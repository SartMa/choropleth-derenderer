# Training on the lab machine (Windows, RTX A4000)

What gets trained: answer sheet v3 (S2 + zoom, `train/answer_v3.py`) **from scratch**: plain Qwen2.5-VL-3B with a
fresh LoRA and a fresh region head, two passes over 3,704 MapQA + 10,000 synthetic US + 10,000 synthetic India maps,
then scored on the full test sets (4,157 MapQA, 1,000 synthetic India, 68 real India maps). The Kaggle v3 model (built
by continuing v1 -> v2 -> v3) is downloaded only to be scored on the same test sets for comparison:
`python lab\run.py score data\lab\inputs\v3-weights`. See `docs/ANSWER_SHEET.md` for the method.

Everything goes through `python lab/run.py <step>`, run from the repo folder. Data, the Qwen download and results go
to `data\lab\` (git-ignored).

The lab machine has **hardware virtualization off in the BIOS**, so Docker Desktop and WSL2 cannot run there. We run
natively on Windows with conda instead (section A). If virtualization is ever turned on (Intel VT-x / AMD SVM in the
BIOS), the Docker route in section B also works.

## A. Native Windows with conda (use this)

### A1. Once: environment and code

In the **Anaconda PowerShell Prompt** (or any PowerShell where `conda` works):

```powershell
conda create -n derender python=3.11 -y
conda activate derender
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
git clone -b answer-sheet https://github.com/SartMa/choropleth-derenderer.git
cd choropleth-derenderer
pip install -r lab\requirements.txt
```

No git on the machine? Download the `answer-sheet` branch as a ZIP from GitHub and unpack it instead.

Then, **every time you open a new window** for this:

```powershell
conda activate derender
cd <path>\choropleth-derenderer
$env:PYTHONUTF8 = "1"
```

Also set Windows to **never sleep** while plugged in (Settings > System > Power). A run that stops can be resumed,
but it loses time.

### A2. Check, download, smoke test

```powershell
python lab\run.py check                          # GPU, versions, speed, free disk (need ~30 GB)
$env:KAGGLE_API_TOKEN = "<the skini501 token>"   # only in this window; never written to a file
python lab\run.py prepare                        # ~10 GB of maps + the Kaggle v3 model, then converts MapQA
python lab\run.py smoke                          # ~30-40 minutes
```

`check` must say the GPU is an RTX A4000 with bf16. If it says no CUDA GPU, the PyTorch install got the CPU build:
run the `pip install torch...` line again.

The smoke test trains on a few maps, **stops itself on purpose** after 6 updates, resumes from the checkpoint,
finishes, keeps a snapshot and scores a few maps. It ends with `smoke PASSED` and an estimate of how long the full
run's training takes on this GPU. The first run also downloads Qwen2.5-VL-3B (7.5 GB) into `data\lab\hf`.

### A3. The full run

Open a separate window for it (A1's three lines), then:

```powershell
python lab\run.py full
```

Leave that window open. From any other window (A1's three lines first):

```powershell
python lab\run.py status          # training update, test maps read, snapshots kept
```

- **Stopping and continuing.** Ctrl+C in the run's window, closing it, a reboot or a power cut all stop the run. To
  continue, run `python lab\run.py full` again. It picks up from the last checkpoint (saved every 30 minutes), in the
  same data order; scoring skips maps already read.
- **Kept copies.** Every 6 hours of training, a usable copy of the model goes to
  `data\lab\runs\full\snapshots\update_NNNNN\` and is never overwritten. To score one:
  `python lab\run.py score data\lab\runs\full\snapshots\update_01234`.
- **Results** are in `data\lab\runs\full\`: `metrics_mapqa.json`, `metrics_india.json`, `predictions_real.jsonl`
  (score with `lookup\score_real.py`), `metrics_head.json`, `train_log.jsonl`, and the final `adapter\` +
  `region_head.pt`. The whole console log is in `run.log`.

Expected time: training about two days on the A4000 (the smoke test prints a better estimate), then several hours
of scoring.

## A4. The tiles model, for comparison

The tiles model (the map plus 2 tiles magnified 3x; a rule places them in training, an MLP picker taught afterwards
places them when reading; `train/train_qwen.py`) is trained the same way: from plain Qwen, on **exactly the same
maps** (the same 3,704 MapQA maps, all 10,000 synthetic US and 10,000 synthetic India), 2 passes, the same batch and
learning rate, scored on the same test sets including the 68 real India maps.

```powershell
python lab\run.py tiles-smoke     # ~30 min, same checks as the answer sheet's smoke test
python lab\run.py tiles-full      # after the answer sheet's full run has finished: one run at a time on the GPU
python lab\run.py status          # shows both runs
```

Its results are in `data\lab\runs\tiles-full\` with `_mlp` in the names (`metrics_mapqa_mlp.json`,
`predictions_real_mlp.jsonl`, ...). Snapshots: `python lab\run.py tiles-score data\lab\runs\tiles-full\snapshots\update_NNNNN`.

## B. Docker (only if hardware virtualization is on)

`docker ps` failing with *"request returned Internal Server Error for API route ... docker_engine"* means Docker
Desktop's engine is not running; on this machine that is because virtualization is off. With it on:

```powershell
docker build -t derender lab
function derender { docker run --rm -it --gpus all --shm-size 16g -v "${PWD}:/work" -e KAGGLE_API_TOKEN derender python lab/run.py @args }
derender check; derender prepare; derender smoke
docker run -d --name derender-full --gpus all --shm-size 16g -v "${PWD}:/work" derender python lab/run.py full
```

## If something goes wrong

- `no CUDA GPU visible`: PyTorch is the CPU build (reinstall with the `--index-url` line), or another program holds
  the GPU.
- `CUDA out of memory`: should not happen (same batch settings as the 16 GB Kaggle T4); close other GPU programs
  (games, browsers with heavy tabs) and run the same command again.
- `UnicodeDecodeError`: `$env:PYTHONUTF8 = "1"` was not set in that window.
- Downloads hang: pause Cloudflare WARP.
- Anything else: send the last 50 lines of `data\lab\runs\full\run.log` (or of the smoke output).
