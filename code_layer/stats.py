"""The spatial statistics, with permutation significance.

Significance comes from randomisation rather than a normal approximation: shuffle the values across
regions many times and see how often chance produces something as extreme as the real map. Every
function takes an explicit seed, because an unseeded permutation test is not reproducible and the
layer's whole claim is that its output is.
"""

import numpy as np

DEFAULT_PERMUTATIONS = 999


def _rng(seed):
    return np.random.default_rng(seed)


def benjamini_hochberg(pvalues):
    """FDR-adjusted p-values. Testing every region at 0.05 would flag ~2-3 of 50 by chance alone."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def morans_i(x, w):
    """Global Moran's I: is the map clustered at all? Around -1/(n-1) when values are scattered."""
    z = x - x.mean()
    denominator = float(z @ z)
    if denominator == 0:
        return 0.0
    return float(len(x) * (z @ (w.matrix @ z)) / (w.matrix.sum() * denominator))


def morans_i_test(x, w, permutations=DEFAULT_PERMUTATIONS, seed=0):
    observed = morans_i(x, w)
    expected = -1.0 / (len(x) - 1) if len(x) > 1 else 0.0
    rng = _rng(seed)
    null = np.array([morans_i(rng.permutation(x), w) for _ in range(permutations)])
    extreme = int((np.abs(null - expected) >= abs(observed - expected)).sum())
    return {"statistic": observed, "expected": expected,
            "p_value": (extreme + 1) / (permutations + 1), "permutations": permutations}


def local_morans(x, w, permutations=DEFAULT_PERMUTATIONS, seed=0, alpha=0.05):
    """Local Moran's I per region, with conditional permutation and an FDR correction.

    Conditional means a region keeps its own value and only its neighbours are reshuffled, which is
    the question being asked: given this region's value, is its neighbourhood unusual?
    """
    n = len(x)
    z = x - x.mean()
    denominator = float(z @ z)
    if denominator == 0:
        zeros = np.zeros(n)
        return {"statistic": zeros, "lag": zeros, "z": z, "p_value": np.ones(n),
                "p_adjusted": np.ones(n), "significant": np.zeros(n, bool),
                "quadrant": ["none"] * n}
    scale = (n - 1) / denominator      # esda's Moran_Local convention, so results are comparable
    lag = w.lag(z)
    statistic = scale * z * lag

    rng = _rng(seed)
    counts = np.zeros(n)
    neighbours = [int(round(row.sum())) for row in w.binary]
    for i in range(n):
        others = np.delete(z, i)
        k = neighbours[i]
        drawn = np.array([rng.choice(others, size=k, replace=False).mean() for _ in range(permutations)])
        simulated = scale * z[i] * drawn
        counts[i] = (np.abs(simulated) >= abs(statistic[i])).sum()
    p = (counts + 1) / (permutations + 1)
    adjusted = benjamini_hochberg(p)

    quadrant = []
    for i in range(n):
        if z[i] > 0 and lag[i] > 0:
            quadrant.append("high_high")
        elif z[i] < 0 and lag[i] < 0:
            quadrant.append("low_low")
        elif z[i] > 0 and lag[i] < 0:
            quadrant.append("high_low")
        elif z[i] < 0 and lag[i] > 0:
            quadrant.append("low_high")
        else:
            quadrant.append("none")
    return {"statistic": statistic, "lag": lag, "z": z, "p_value": p, "p_adjusted": adjusted,
            "significant": adjusted <= alpha, "quadrant": quadrant}


def getis_ord(x, w, permutations=DEFAULT_PERMUTATIONS, seed=0, alpha=0.05):
    """Getis-Ord Gi*: how intense is each region's neighbourhood, counting the region itself.

    Used to rank clusters by strength once Local Moran has said which regions form them.

    This is the difference form of Ord & Getis (1995), the one ArcGIS implements. esda uses the ratio
    form instead, which normalises by the sum of all values; the two rank regions almost identically
    but their z-scores are on different scales, so do not compare the numbers across the two.
    """
    n = len(x)
    binary = w.binary + np.eye(n)          # Gi* includes the region itself
    totals = binary.sum(1)
    mean, sigma = x.mean(), x.std()
    if sigma == 0:
        return {"z_score": np.zeros(n), "p_value": np.ones(n), "p_adjusted": np.ones(n),
                "significant": np.zeros(n, bool)}
    numerator = binary @ x - mean * totals
    denominator = sigma * np.sqrt((n * totals - totals ** 2) / (n - 1))
    z_score = np.divide(numerator, denominator, out=np.zeros(n), where=denominator != 0)

    rng = _rng(seed)
    counts = np.zeros(n)
    for _ in range(permutations):
        shuffled = rng.permutation(x)
        simulated = (binary @ shuffled - mean * totals)
        simulated = np.divide(simulated, denominator, out=np.zeros(n), where=denominator != 0)
        counts += np.abs(simulated) >= np.abs(z_score)
    p = (counts + 1) / (permutations + 1)
    adjusted = benjamini_hochberg(p)
    return {"z_score": z_score, "p_value": p, "p_adjusted": adjusted, "significant": adjusted <= alpha}


def trend_surface(x, coords, w, permutations=DEFAULT_PERMUTATIONS, seed=0):
    """OLS of value on equal-area centroid x/y, with a check on the residuals.

    OLS standard errors assume independent residuals. On a spatially clustered map they are not, and
    the p-values come out too small, so the residuals are tested for autocorrelation and the result
    is reported alongside rather than quietly ignored.
    """
    n = len(x)
    design = np.column_stack([np.ones(n), coords[:, 0], coords[:, 1]])
    coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ coefficients
    residuals = x - fitted
    dof = n - design.shape[1]
    if dof <= 0:
        return None
    sigma2 = float(residuals @ residuals) / dof
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))

    rng = _rng(seed)
    observed = np.abs(coefficients[1:] / np.where(errors[1:] == 0, np.inf, errors[1:]))
    counts = np.zeros(2)
    for _ in range(permutations):
        shuffled = rng.permutation(x)
        beta, *_ = np.linalg.lstsq(design, shuffled, rcond=None)
        resid = shuffled - design @ beta
        s2 = float(resid @ resid) / dof
        err = np.sqrt(np.diag(s2 * np.linalg.pinv(design.T @ design)))
        counts += np.abs(beta[1:] / np.where(err[1:] == 0, np.inf, err[1:])) >= observed
    p = (counts + 1) / (permutations + 1)
    residual_test = morans_i_test(residuals, w, permutations=199, seed=seed + 1)
    return {"east_west": float(coefficients[1]), "south_north": float(coefficients[2]),
            "p_east_west": float(p[0]), "p_south_north": float(p[1]),
            "r_squared": float(1 - (residuals @ residuals) / ((x - x.mean()) @ (x - x.mean())))
            if x.var() > 0 else 0.0,
            "residual_moran_p": residual_test["p_value"]}
