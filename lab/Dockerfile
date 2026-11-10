# Only for machines with hardware virtualization (Docker Desktop needs it); otherwise use the conda steps in
# lab/README.md. Same library versions as the Kaggle runs, on CUDA 12.4. The repo is mounted at /work, not copied.
#
#   docker build -t derender lab
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /work
