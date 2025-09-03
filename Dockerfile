
FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive


ENV TZ=UTC


RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl gettext-base \
    openslide-tools ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt setup.py ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["dvc", "repro"]
