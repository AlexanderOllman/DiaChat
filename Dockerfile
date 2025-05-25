FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---- system deps ----
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential python3 python3-venv python3-pip ffmpeg libportaudio2 libsndfile1 pulseaudio openssl && \
    rm -rf /var/lib/apt/lists/*

# ---- python environment ----
ENV PYTHONUNBUFFERED=1
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

# ---- build whisper.cpp ----
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp && \
    cd whisper.cpp && make -j$(nproc) WHISPER_CUDA=1 WHISPER_FAST=1 && \
    cp libwhisper.a /venv/lib && cd ..

# ---- install Dia and dependencies ----
RUN pip install torch==2.3.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html && \
    git clone https://github.com/nari-labs/dia.git && pip install -e dia

# ---- install app deps ----
RUN pip install \
        ollama httpx sounddevice webrtcvad pywhispercpp \
        fastapi uvicorn[standard] resampy scipy huggingface_hub

# ---- copy source ----
WORKDIR /app
COPY main.py index.html setup_models.py ./

# ---- persistent models volume ----
VOLUME ["/models"]
ENV MODELS_DIR=/models

# ---- fetch models at build time ----
RUN python setup_models.py

EXPOSE 8000
ENTRYPOINT ["python", "main.py"]