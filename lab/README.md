# Training on the lab machine (Windows, RTX A4000)

Everything runs in a Linux Docker container on the GPU, with the same library versions as our Kaggle runs. The code
stays in the repo folder and is mounted into the container; data, the Qwen download and results go to `data\lab\`
(git-ignored). Commands below are for PowerShell, run from the repo folder.

What gets trained: answer sheet v3 (S2 + zoom, `train/answer_v3.py`), continuing from the Kaggle v3 weights, on
3,704 MapQA + 8,000 synthetic US + 10,000 synthetic India maps (one full pass), then scored on the full test sets
(4,157 MapQA, 1,000 synthetic India, 68 real India maps). See `docs/ANSWER_SHEET.md` for the method.

## 0. Once: get Docker's engine running

`docker --version` working but `docker ps` failing with *"request returned Internal Server Error for API route and
version ... docker_engine"* means the Docker command is installed but **Docker Desktop's engine is not running**.

1. Start **Docker Desktop** from the Start menu and wait until it shows *Engine running* (green, bottom left).
   Then `docker ps` should print an empty table.
2. If it never starts: in PowerShell **as Administrator** run `wsl --update`, then `wsl --status` (default version
   should be 2), restart Docker Desktop. In Docker Desktop, Settings > General, *Use the WSL 2 based engine* must be on.
3. If downloads hang later, try pausing Cloudflare WARP (it is running on this machine and can block Docker's network).
4. Check the GPU works inside a container:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   It should print the same RTX A4000 table as `nvidia-smi` on Windows.

Also set Windows to **never sleep** while plugged in (Settings > System > Power), or a long run stops when the
machine sleeps (it can be resumed, but it loses time).

## 1. Once: get the code and build the image

```powershell
git clone -b answer-sheet https://github.com/SartMa/choropleth-derenderer.git
cd choropleth-derenderer
docker build -t derender lab
```

The build downloads about 4 GB (PyTorch with CUDA 12.4) and takes 5-15 minutes.

To save typing, define this once per PowerShell window (it starts the container on the GPU with the repo mounted):

```powershell
function derender { docker run --rm -it --gpus all --shm-size 16g -v "${PWD}:/work" -e KAGGLE_API_TOKEN derender python lab/run.py @args }
```

## 2. Check, download, smoke test

```powershell
derender check                                  # GPU, versions, matmul speed, free disk (need ~30 GB)
$env:KAGGLE_API_TOKEN = "<the skini501 token>"  # only in this window; never written to a file
derender prepare                                # ~10 GB of maps + v3's weights, then converts MapQA
derender smoke                                  # ~30 minutes
```

The smoke test trains on a few maps, **stops itself on purpose** after 6 updates, resumes from the checkpoint,
finishes, keeps a snapshot and scores a few maps. It ends with `smoke PASSED` and an estimate of how long the full
run's training will take on this GPU. The first run also downloads Qwen2.5-VL-3B (7.5 GB) into `data\lab\hf`.

## 3. The full run

Run it detached, so closing the window does not stop it:

```powershell
docker run -d --name derender-full --gpus all --shm-size 16g -v "${PWD}:/work" derender python lab/run.py full
docker logs -f derender-full          # watch it (Ctrl+C stops watching, not the run)
derender status                       # progress: training update, test maps read, snapshots kept
```

- **Stopping and continuing.** `docker stop derender-full` (or a reboot, a power cut, someone needing the GPU). To
  continue: `docker rm derender-full` and the same `docker run -d ...` line again. It picks up from the last
  checkpoint (saved every 30 minutes), in the same data order, and scoring skips maps already read.
- **Kept copies.** Every 6 hours of training, a usable copy of the model goes to
  `data\lab\runs\full\snapshots\update_NNNNN\` and is never overwritten. To score one:
  `derender score data/lab/runs/full/snapshots/update_01234`.
- **Results** are in `data\lab\runs\full\`: `metrics_mapqa.json`, `metrics_india.json`, `predictions_real.jsonl`
  (score with `lookup/score_real.py`), `metrics_head.json`, `train_log.jsonl`, and the final `adapter\` +
  `region_head.pt`. The whole console log is in `run.log`.

Expected time: training about a day on the A4000 (the smoke test prints a better estimate), then several hours of
scoring.

## If something goes wrong

- `no CUDA GPU visible`: the container was started without `--gpus all`, or step 0.4 fails.
- `CUDA out of memory`: should not happen (same batch settings as the 16 GB Kaggle T4); if it does, close other GPU
  programs and continue the run.
- A DataLoader error about shared memory: the `--shm-size 16g` part is missing.
- Anything else: send the last 50 lines of `data\lab\runs\full\run.log` (or of the smoke output).
