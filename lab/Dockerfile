# The training environment for the lab machine: the same library versions as the Kaggle runs, on CUDA 12.4
# (the lab's NVIDIA driver 551.86 supports up to 12.4). The code is not copied in: the repo is mounted at /work.
#
#   docker build -t derender lab
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "transformers==4.57.1" "peft==0.17.1" "accelerate>=1.0,<2" "kaggle==2.2.4" \
        "opencv-python-headless==4.10.0.84" scipy pillow
WORKDIR /work
