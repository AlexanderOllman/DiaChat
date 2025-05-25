# DiaChat

DiaChat is a lightweight, real-time, in-browser voice AI chat framework designed to be run locally. It uses a combination of cutting-edge speech-to-text (STT) from [Whisper](https://github.com/ggerganov/whisper.cpp), a large language model (LLM) via [Ollama](https://github.com/ollama/ollama), and text-to-speech (TTS) with the [Dia](https://github.com/nari-labs/dia) model to provide a conversational AI experience. 

## Features

- **Run it anywhere!** Designed to be run on lightweight hardware. Written and tested on an RTX 3090.
- **Real-time Voice Interaction:** Speak to through a browser (microphone enabled), remote client or on-device.
- **High-Quality STT:** Powered by Whisper (via `pywhispercpp`) for fast, accurate transcription.
- **Advanced LLM:** Leverages your existing Ollama instance to let you configurable your own language model (defaults to Llama 3.2) for intelligent and contextual responses.
- **Natural TTS:** Employs the Dia model (`nari-labs/Dia-1.6B`) for expressive and human-like speech synthesis.
- **Web-Based UI:** Simple and intuitive interface built with FastAPI and HTML/JavaScript, featuring:
    - Start/Stop controls
    - Status indicators (idle, listening, processing, speaking)
    - Live audio waveform visualization
    - Display of transcribed text (Whisper output)
    - Display of TTS processing status
    - Chat log
    - Playback speed control for AI responses
- **CUDA Acceleration:** Supports GPU acceleration for faster model inference (if a compatible NVIDIA GPU is available).
- **Dockerized:** Comes with a `Dockerfile` for easy setup and deployment.
- **Model Management:** Includes a script (`setup_models.py`) to download and cache required AI models.

## Architecture

DiaChat consists of the following main components:

1.  **Frontend (`index.html`):** Handles user interaction, captures microphone audio, displays chat messages, and plays back AI-generated speech.
2.  **Backend (`main.py`):**
    *   A FastAPI application that serves the frontend and manages WebSocket connections for audio streaming and communication.
    *   **STT Worker:** Transcribes incoming audio from the user using Whisper.
    *   **LLM Worker:** Processes the transcribed text using an Ollama-hosted LLM to generate a response.
    *   **TTS Worker:** Converts the LLM's text response into speech using the Dia model.
    *   Pre-flight checks to ensure all model dependencies and services are correctly configured.
3.  **Model Setup (`setup_models.py`):** A utility script to download and cache the Whisper and Dia models from Hugging Face.
4.  **Containerization (`Dockerfile`):** Defines the environment and steps to build a Docker image for the application, including installing dependencies, building Whisper.cpp with CUDA support, and setting up models.

## Tech Stack

-   **Python:** Core application logic.
-   **FastAPI:** Web framework for the backend API and WebSocket communication.
-   **Uvicorn:** ASGI server for FastAPI.
-   **Whisper (`pywhispercpp`):** Speech-to-text.
-   **Ollama (with Llama 3.2 or other models):** Large Language Model for response generation.
-   **Dia (`nari-labs/Dia-1.6B`):** Text-to-speech.
-   **WebRTC VAD:** Voice Activity Detection.
-   **Resampy:** Audio resampling.
-   **NumPy:** Numerical operations.
-   **Hugging Face Hub:** For model downloading.
-   **Docker:** For containerization.
-   **HTML, CSS, JavaScript:** Frontend user interface.

## Prerequisites

-   Docker
-   NVIDIA GPU with CUDA drivers (recommended for optimal performance, but will fallback to CPU)
-   Access to an Ollama instance (running locally or remotely). Ensure the model specified in `main.py` (default: `llama3.2:latest`) is available or can be pulled by Ollama.
-   `openssl` command-line tool (for automatic self-signed certificate generation if needed).

## Setup and Running

### 1. Build and Run

#### 1.1 Running Locally (without Docker)

If you prefer to run the application directly on your host machine without Docker:

1.  **Prerequisites:**
    *   Python 3.9+ and `pip`
    *   Git
    *   An Ollama instance running and accessible (see Ollama documentation for setup).
    *   System dependencies for `pywhispercpp` (which uses `whisper.cpp`): `build-essential`, `ffmpeg` (or equivalent for your OS). For `whisper.cpp` CUDA support, you'll need the CUDA toolkit installed matching your NVIDIA drivers.
    *   System dependencies for audio playback/recording if not already present (e.g., `libportaudio2`, `libsndfile1`, `pulseaudio` on Debian/Ubuntu).
    *   `openssl` command-line tool (for automatic self-signed certificate generation if needed).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AlexanderOllman/DiaChat
    cd diachat
    ```

3.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    *   **PyTorch with CUDA (Recommended for GPU):**
        Before running `pip install -r requirements.txt`, install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). For example:
        ```bash
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 # Or your CUDA version
        ```
    *   **Install whisper.cpp Python Bindings and Other Dependencies:**
        The `pywhispercpp` package in `requirements.txt` will attempt to build `whisper.cpp` when installed. Ensure you have its build dependencies.
        ```bash
        pip install -r requirements.txt
        ```
        Note: The `requirements.txt` installs `dia` directly from its Git repository.

5.  **Set Environment Variables (Optional but Recommended):**
    Configure the application using environment variables. It's good practice to set `MODELS_DIR` to a persistent location.
    ```bash
    export MODELS_DIR=$(pwd)/models # Or any other path
    export OLLAMA_HOST="http://localhost:11434" # If Ollama is local
    export OLLAMA_MODEL="llama3.2:latest"
    # HF_HUB_CACHE will be set to MODELS_DIR by setup_models.py
    ```
    Create the models directory if it doesn't exist: `mkdir -p $MODELS_DIR`

6.  **Download AI Models:**
    Run the `setup_models.py` script. This will use the `MODELS_DIR` environment variable if set.
    ```bash
    python3 setup_models.py
    ```
    This script may prompt for Hugging Face login if the Dia model requires authentication.

7.  **SSL Certificate Generation (for Microphone Access on Non-Localhost URLs):**
    For microphone access in the browser when accessing the server using an IP address (e.g., `https://192.168.x.x:8443`) rather than `http://localhost:8000` or `https://localhost:8443`, a self-signed SSL certificate is usually required. Browsers enforce HTTPS for microphone access on non-localhost URLs for security reasons.

    During the initial **Pre-flight Checks** (see step 3 below), the application (`main.py`) will automatically attempt to generate `cert.pem` and `key.pem` files if they are not found in the project's root directory and the `openssl` command is available on your system.
    
    If certificates are missing and `openssl` is found, you will be prompted to enter a **Common Name (CN)** for the certificate during the pre-flight check process. This is crucial:
    *   If you will only access DiaChat via `localhost`, you can enter `localhost` or press Enter to use it as the default.
    *   If you plan to access DiaChat from other devices on your local network using your machine's IP address (e.g., `192.168.1.100`), you **must** enter that IP address as the Common Name when prompted.
    
    The other fields in the certificate subject (Country, State, etc.) will be filled with placeholder values. If the automatic generation fails, `openssl` is not found, or you skip the prompt, the application will inform you and will attempt to start with HTTP. You may need to generate the certificate manually using a command like this in the project root if HTTPS is desired and auto-generation failed:
    ```bash
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=YOUR_CHOSEN_COMMON_NAME"
    ```
    Replace `YOUR_CHOSEN_COMMON_NAME` accordingly (e.g., `localhost` or your IP address).

    When you access the site with this self-signed certificate, your browser will show a security warning that you'll need to accept/bypass.

8.  **Run the Application:**
    ```bash
    python3 main.py
    ```
    The application will perform pre-flight checks. If successful, it will start the Uvicorn server. By default, it tries to run an HTTPS server on port 8443 if `cert.pem` and `key.pem` are found in the root directory. Otherwise, it falls back to HTTP on port 8000.

    For microphone access in the browser, you often need to serve the site over HTTPS or access it via `http://localhost` (not `http://127.0.0.1`). If using HTTP and accessing from a different device on your network, microphone access might be blocked by the browser.

    You can skip pre-flight checks for faster startup (if you know everything is set up):
    ```bash
    python3 main.py --skip-checks
    ```

#### 1.2 Using Docker

The `Dockerfile` handles most of the setup for a containerized deployment.

1.  **Build the Docker Image:**
    (Ensure you have cloned the repository first as shown in the "Running Locally" section, step 2, if not already done.)
    ```bash
    docker build -t diachat .
    ```

2.  **Run the Docker Container:**
    (Configure environment variables as needed, similar to step 5 in "Running Locally", by passing them with `-e VAR=value` to `docker run` or using a `.env` file if your Docker setup supports it. `MODELS_DIR` is set to `/models` inside the container by the Dockerfile.)
    ```bash
    docker run -p 8000:8000 --gpus all -v $(pwd)/models:/models diachat
    ```

    **Explanation of `docker run` flags:**
    *   `-p 8000:8000`: Maps port 8000 on your host to port 8000 in the container.
    *   `--gpus all`: Enables GPU access for the container (requires NVIDIA Container Toolkit). If you don't have a GPU or NVIDIA Container Toolkit, you can try removing this flag, and the application will attempt to run on CPU (performance will be significantly impacted).
    *   `-v $(pwd)/models:/models`: Mounts a local directory named `models` into the `/models` directory in the container. This allows models to be persisted across container restarts. Create this directory if it doesn't exist: `mkdir models`.

    **Note on Hugging Face Authentication (for Docker build):**
    The `setup_models.py` script (run during `docker build`) will attempt to download models. The Dia model, in particular, might require authentication with Hugging Face. The script will prompt for login if needed during the build process. If you prefer, you can log in to Hugging Face CLI on your host machine (`huggingface-cli login`) and then mount your Hugging Face token and cache into the container during the run command (though this is more for ensuring the *run-time* `setup_models.py` if manually invoked inside container works, the build itself needs auth if models are gated):
    
    If the build fails due to auth, ensure you are logged in with `huggingface-cli login` where the `docker build` command is being executed, or pre-download models to a directory and mount it if the build process itself cannot authenticate interactively. For running the container with pre-existing `setup_models.py` which might re-check or download:
    ```bash
    docker run -p 8000:8000 --gpus all \
        -v $(pwd)/models:/models \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        diachat
    ```

### 2. Access DiaChat

Open your web browser and navigate to `http://localhost:8000` (or `https://localhost:8443` if using HTTPS, for local non-Docker runs). For Docker, it's typically `http://localhost:8000`.

### 3. Pre-flight Checks

Upon starting (either locally or in Docker), the application runs a series of pre-flight checks to ensure all components (Audio Processing, Whisper, Dia, Ollama, SSL Certificates) are working correctly or to attempt necessary setup (like SSL certificate generation).

Check the application logs (and the browser console for any frontend issues) if you encounter problems. The SSL check will inform you if certificates were found, generated, or if generation failed (in which case HTTP will be used).

If model-related checks fail, you might need to:
*   Ensure the `models` directory exists and is writable by the application.
*   Verify that the `whisper.cpp` and `dia` executables are correctly installed and accessible.
*   Check the system logs for any errors related to model loading or inference.

If the SSL check fails and you need to manually generate certificates, you can use the following command in the project root directory:
```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=YOUR_CHOSEN_COMMON_NAME"
```
Replace `YOUR_CHOSEN_COMMON_NAME` accordingly (e.g., `localhost` or your IP address).

When you access the site with this self-signed certificate, your browser will show a security warning that you'll need to accept/bypass.