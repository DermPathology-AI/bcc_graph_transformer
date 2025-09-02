FROM registry.git.vgregion.se/aiplattform/images/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Stockholm/Europe

RUN apt -y update \
    && apt install python3 python3-pip git -y \
    && apt install python-is-python3 -y \
    && apt-get install -y curl gettext-base \
    && apt-get install -y  openslide-tools \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get install -y libgl1-mesa-glx \
    && addgroup --gid 1000 researcher \
    && adduser --home /workspace --disabled-password --gecos '' --uid 1000 --gid 1000 researcher

COPY requirements.txt setup.py ./
RUN pip install -r requirements.txt

USER researcher
WORKDIR /workspace

CMD ["dvc", "repro"]
