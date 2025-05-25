import os
import asyncio
import json
import queue
import threading
import ssl
import sys
import time
import argparse
import subprocess
import shutil

import numpy as np
import resampy
import webrtcvad
import httpx
import ollama
from pywhispercpp.model import Model
from dia.model import Dia
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import torch

# ---------- configuration ----------
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
# Set Hugging Face cache directory to match setup_models.py
os.environ["HF_HUB_CACHE"] = os.path.abspath(MODELS_DIR)

WHISPER_MODEL_NAME = "base.en"  # pywhispercpp will use the local model if available
DIA_MODEL_NAME = "nari-labs/Dia-1.6B"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# DIA_SYSTEM_PROMPT = """You are a helpful and conversational AI assistant. Your responses will be converted into speech by an advanced Text-to-Speech (TTS) system. To make the speech sound more natural, please try to incorporate the following:
# 1. Where it feels appropriate and natural, include non-verbal sounds. The TTS system understands cues such as (clears throat), (sighs), (chuckle), (laughs), (gasps). Please use them sparingly to enhance realism. For example: '(clears throat) I believe the answer is...' or 'That's quite funny! (chuckle)'.
# 2. Use natural language phrasing to create pauses or hesitations where they would normally occur in conversation. For example, you can use phrases like "Well...", "Hmm...", or use ellipses like "...". The TTS system does not use a special '(pause)' tag, so rely on these natural linguistic cues.
# 3. Please keep your responses coherent, directly answer the user's query, and maintain a friendly conversational tone.
# """

DIA_SYSTEM_PROMPT = """You are a helpful and conversational AI assistant. Your responses will be converted into speech by an advanced Text-to-Speech (TTS) system. To make the speech sound more natural, please keep your responses coherent, directly answer the user's query, and maintain a friendly conversational tone."""

# Auto-detect device (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[main] Using device: {DEVICE}")

# ---------- Pre-flight checks ----------

def check_or_generate_ssl_certificate():
    """Checks for SSL certs, tries to generate them if missing and openssl is available."""
    cert_file = "cert.pem"
    key_file = "key.pem"
    print("[check] Checking SSL Certificates...")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"[check] ✅ SSL certificates ({cert_file}, {key_file}) found.")
        return True

    print(f"[check] ⚠️ SSL certificate files ({cert_file}, {key_file}) not found.")
    if not shutil.which("openssl"):
        print("[check] ❌ 'openssl' command not found. Cannot generate certificate automatically.")
        print(f"[check] For HTTPS, please generate '{cert_file}' and '{key_file}' manually or install openssl.")
        print("[check] Application will attempt to start with HTTP.")
        return True # Not a fatal error for pre-flight, allows HTTP fallback

    print("[check] Attempting to generate self-signed SSL certificate via OpenSSL...")
    common_name = input(f"[check] Enter Common Name (CN) for the certificate (e.g., localhost, or your server's IP like 192.168.x.x): ").strip()
    if not common_name:
        common_name = "localhost" # Default if empty
        print(f"[check] No Common Name provided, defaulting to '{common_name}'.")

    subj = f"/C=US/ST=State/L=City/O=Organization/CN={common_name}"
    openssl_command = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
        "-out", cert_file, "-keyout", key_file,
        "-days", "365", "-subj", subj
    ]
    try:
        print(f"[check] Running command: {' '.join(openssl_command)}")
        process = subprocess.run(openssl_command, capture_output=True, text=True, check=True)
        print(f"[check] ✅ SSL certificate generated successfully: {cert_file}, {key_file}")
        os.chmod(cert_file, 0o644)
        os.chmod(key_file, 0o600)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[check] ❌ Error generating SSL certificate with OpenSSL:")
        print(f"[check] STDOUT: {e.stdout}")
        print(f"[check] STDERR: {e.stderr}")
    except FileNotFoundError: # Should be caught by shutil.which, but as a safeguard
         print(f"[check] ❌ Error: 'openssl' command not found during execution attempt.")
    except Exception as e:
        print(f"[check] ❌ An unexpected error occurred during certificate generation: {e}")
    
    print("[check] SSL certificate generation failed. Application will attempt to start with HTTP.")
    return True # Not a fatal error, allow HTTP fallback

def check_ollama():
    """Test Ollama connectivity and model availability"""
    print("[check] Testing Ollama connection...")
    try:
        # Create client with host
        client = ollama.Client(host=OLLAMA_HOST)
        
        # Test basic connectivity
        response = client.list()
        # Handle Ollama's Model objects which have .model attribute, not ['name']
        models_list = response.get('models', [])
        available_models = []
        for model in models_list:
            if hasattr(model, 'model'):
                available_models.append(model.model)
            elif isinstance(model, dict) and 'name' in model:
                available_models.append(model['name'])
            else:
                print(f"[check] Warning: Unexpected model format: {model}")
        
        print(f"[check] ✅ Ollama connected. Available models: {available_models}")
        
        # Check if our required model is available
        if OLLAMA_MODEL not in available_models:
            print(f"[check] ⚠️  Model '{OLLAMA_MODEL}' not found. Attempting to pull...")
            try:
                client.pull(OLLAMA_MODEL)
                print(f"[check] ✅ Successfully pulled {OLLAMA_MODEL}")
            except Exception as e:
                print(f"[check] ❌ Failed to pull {OLLAMA_MODEL}: {e}")
                return False
        
        # Test model inference
        print(f"[check] Testing {OLLAMA_MODEL} inference...")
        test_response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Hello, are you working?"}]
        )
        response_text = test_response['message']['content']
        print(f"[check] ✅ Ollama test successful. Response: {response_text[:50]}...")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Ollama connection failed: {e}")
        print(f"[check] Make sure Ollama is running at {OLLAMA_HOST}")
        return False

def check_whisper():
    """Test Whisper model loading"""
    print("[check] Testing Whisper model...")
    try:
        model = Model(WHISPER_MODEL_NAME)
        print(f"[check] ✅ Whisper model '{WHISPER_MODEL_NAME}' loaded successfully")
        
        # Test with a small audio sample (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
        segments = model.transcribe(test_audio)
        print(f"[check] ✅ Whisper inference test successful")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Whisper model failed: {e}")
        print(f"[check] Make sure the model is downloaded. Run: python3 setup_models.py")
        return False

def check_dia():
    """Test Dia model loading and inference"""
    print("[check] Testing Dia model...")
    try:
        dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
        print(f"[check] ✅ Dia model loaded successfully on {DEVICE}")
        
        # Test with a short text
        print("[check] Testing Dia inference...")
        test_text = "[S1] Hello, this is a test."
        audio_output = dia.generate(test_text)
        print(f"[check] ✅ Dia inference successful. Generated {audio_output.shape} audio samples")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Dia model failed: {e}")
        print(f"[check] Make sure you're logged into Hugging Face and the model is cached")
        return False

def check_audio_processing():
    """Test audio processing components"""
    print("[check] Testing audio processing...")
    try:
        # Test webrtcvad
        vad = webrtcvad.Vad(2)
        test_frame = np.zeros(480, dtype=np.int16).tobytes()  # 30ms frame at 16kHz
        vad.is_speech(test_frame, 16000)
        print("[check] ✅ WebRTC VAD working")
        
        # Test resampling
        test_audio = np.random.random(4096).astype(np.float32)
        resampled = resampy.resample(test_audio, 48000, 16000)
        print("[check] ✅ Audio resampling working")
        
        return True
        
    except Exception as e:
        print(f"[check] ❌ Audio processing failed: {e}")
        return False

def run_preflight_checks():
    """Run all pre-flight checks"""
    print("\n🚀 Starting DiaChat pre-flight checks...\n")
    
    checks = [
        ("Audio Processing", check_audio_processing),
        ("Whisper Model", check_whisper),
        ("Dia Model", check_dia),
        ("Ollama Service", check_ollama),
        ("SSL Certificate", check_or_generate_ssl_certificate)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"Checking: {name}")
        print('='*50)
        results[name] = check_func()
        time.sleep(0.5)  # Small delay between checks
    
    # Summary
    print(f"\n{'='*50}")
    print("PRE-FLIGHT RESULTS SUMMARY")
    print('='*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print('='*50)
    
    if all_passed:
        print("🎉 All checks passed! DiaChat is ready to start.\n")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above before starting DiaChat.")
        print("💡 Try running 'python3 setup_models.py' to fix model issues.\n")
        return False

# ---------- FastAPI ---------
app = FastAPI()
clients = set()

# Queues for threading
to_llm = queue.Queue(maxsize=10)
text_queue = queue.Queue(maxsize=10)
to_tts = queue.Queue(maxsize=10)
ws_out_queue = queue.Queue(maxsize=50)  # Queue for WebSocket outbound messages

@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    
    # Start WebSocket message handler for this client
    async def handle_outbound_messages():
        while True:
            try:
                # Check for messages to send to this client
                while not ws_out_queue.empty():
                    try:
                        message_parts = ws_out_queue.get_nowait()
                        msg_type = message_parts[0]
                        payload = message_parts[1]
                        
                        if msg_type == "text":
                            await ws.send_json({"type": "text", "payload": payload})
                        elif msg_type == "whisper_status":
                            processing = message_parts[2] if len(message_parts) > 2 else False
                            await ws.send_json({"type": "whisper_status", "payload": payload, "processing": processing})
                        elif msg_type == "tts_status": # Handle new TTS status message type
                            processing = message_parts[2] if len(message_parts) > 2 else False
                            await ws.send_json({"type": "tts_status", "payload": payload, "processing": processing})
                        elif msg_type == "ai_response":
                            await ws.send_json({"type": "ai_response", "payload": payload})
                        elif msg_type == "audio":
                            await ws.send_bytes(payload)
                    except queue.Empty:
                        break
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            except Exception as e:
                print(f"[ws] Error handling outbound messages: {e}")
                break
    
    # Start the message handler task
    handler_task = asyncio.create_task(handle_outbound_messages())
    
    try:
        while True:
            msg = await ws.receive_bytes()
            pcm = np.frombuffer(msg, np.int16)
            pcm16 = resampy.resample(pcm.astype(np.float32), 48000, 16000)
            to_llm.put(pcm16.astype(np.int16).tobytes())
    except WebSocketDisconnect:
        clients.remove(ws)
        handler_task.cancel()

# ---------- background workers ----------

def stt_worker():
    RATE = 16000
    FRAME_DURATION_MS = 30  # 30ms frames for webrtcvad
    FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  # 480 samples for 30ms at 16kHz
    FRAME_BYTES = FRAME_SIZE * 2  # 2 bytes per int16 sample
    
    vad = webrtcvad.Vad(2)
    model = Model(WHISPER_MODEL_NAME)
    
    audio_buffer = b""
    speech_frames = []
    silence_count = 0
    last_status_time = 0
    
    # Send initial status
    ws_out_queue.put(("whisper_status", "Whisper model loaded and ready", False))
    
    while True:
        buf = to_llm.get()
        audio_buffer += buf
        
        # Process complete frames
        while len(audio_buffer) >= FRAME_BYTES:
            frame = audio_buffer[:FRAME_BYTES]
            audio_buffer = audio_buffer[FRAME_BYTES:]
            
            try:
                is_speech = vad.is_speech(frame, RATE)
                current_time = time.time()
                
                if is_speech:
                    speech_frames.append(frame)
                    silence_count = 0
                    
                    # Send status update every 0.5 seconds during speech
                    if current_time - last_status_time > 0.5:
                        audio_duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                        ws_out_queue.put(("whisper_status", f"🎤 Capturing speech... ({audio_duration:.1f}s)", False))
                        last_status_time = current_time
                        
                else:
                    silence_count += 1
                    
                # If we have speech and then silence, process the speech
                if speech_frames and silence_count > 10:  # ~300ms of silence
                    # Combine all speech frames
                    speech_audio = b"".join(speech_frames)
                    audio_duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                    
                    # Convert bytes to numpy array for pywhispercpp
                    audio_data = np.frombuffer(speech_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Only transcribe if we have enough audio (at least 0.5 seconds)
                    if len(audio_data) > RATE * 0.5:
                        # Send processing status
                        ws_out_queue.put(("whisper_status", f"🔄 Processing {audio_duration:.1f}s of audio...", True))
                        
                        segments = model.transcribe(audio_data)
                        # Extract text from segments
                        txt = " ".join([segment.text for segment in segments]).strip()
                        
                        if txt and len(txt) > 2:  # Only send meaningful text
                            # Send transcription result
                            ws_out_queue.put(("whisper_status", f"✅ Transcribed: '{txt}'", False))
                            # Send to WebSocket clients via queue
                            ws_out_queue.put(("text", txt))
                            # Send to LLM
                            text_queue.put(txt)
                        else:
                            ws_out_queue.put(("whisper_status", "⚠️ No clear speech detected", False))
                    else:
                        ws_out_queue.put(("whisper_status", f"⚠️ Audio too short ({audio_duration:.1f}s), ignoring", False))
                    
                    # Reset for next speech segment
                    speech_frames = []
                    
                    # Reset to listening status after a brief pause
                    threading.Timer(0.5, lambda: ws_out_queue.put(("whisper_status", "👂 Listening for speech...", False))).start()
                    last_status_time = current_time
                    
            except Exception as e:
                # Skip problematic frames
                print(f"[stt] Error processing frame: {e}")
                ws_out_queue.put(("whisper_status", f"❌ Error: {str(e)}", False))
                continue

def llm_worker():
    client = ollama.Client(host=OLLAMA_HOST)
    messages = []
    
    while True:
        user_part = text_queue.get()

        if not messages: # If messages list is empty, it's the start of the conversation
            messages.append({"role": "system", "content": DIA_SYSTEM_PROMPT})
        
        messages.append({"role": "user", "content": user_part})
        
        full_response_content = ""
        sentence_buffer = ""

        for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
            token = chunk["message"]["content"]
            full_response_content += token
            sentence_buffer += token
            
            # Check for sentence termination on the entire buffer
            if any(sentence_buffer.strip().endswith(p) for p in ['.', '?', '!']):
                current_sentence_text = sentence_buffer.strip()
                if current_sentence_text: # Ensure it's not just whitespace or empty
                    to_tts.put(current_sentence_text)
                sentence_buffer = "" # Reset for the next sentence
        
        # Handle any remaining part of the sentence buffer after the stream ends
        if sentence_buffer.strip(): # If the stream ended without punctuation for the last part
            to_tts.put(sentence_buffer.strip())
            # sentence_buffer = "" # Not strictly necessary here as it will be reset on next loop pass

        # Send the complete AI response to the frontend for logging
        if full_response_content.strip():
            ws_out_queue.put(("ai_response", full_response_content.strip()))
            messages.append({"role": "assistant", "content": full_response_content.strip()})
        else:
            # If LLM gave an empty response, still add a placeholder to messages to maintain turn structure
            messages.append({"role": "assistant", "content": "(No audible response)"})

def tts_worker():
    dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
    ws_out_queue.put(("tts_status", "🤖 Dia TTS model loaded and ready", False))
    
    while True:
        sentence = to_tts.get()
        if not sentence:
            continue
        
        # Prepend [S1] for Dia TTS
        dia_input_text = f"[S1] {sentence}"
            
        ws_out_queue.put(("tts_status", f"🔄 Generating audio for: '{dia_input_text[:30]}...'", True))
        
        try:
            # Assuming dia.generate directly returns a NumPy array or similar
            # Remove .cpu().numpy() if it's not a PyTorch tensor output
            pcm24 = dia.generate(dia_input_text, use_torch_compile=True)
            
            # If pcm24 is a PyTorch tensor, ensure it's moved to CPU before numpy conversion
            if hasattr(pcm24, 'cpu'): 
                pcm24 = pcm24.cpu().numpy()
            elif not isinstance(pcm24, np.ndarray):
                # If it's some other type, try to convert to numpy array directly
                # This might need adjustment based on actual type from dia.generate
                pcm24 = np.array(pcm24) 

            pcm48 = resampy.resample(pcm24.astype(np.float32), 24000, 48000) # Ensure float32 for resample
            pcm48_i16 = (pcm48 * 32767).astype(np.int16).tobytes()
            
            ws_out_queue.put(("tts_status", f"✅ Audio generated for: '{dia_input_text[:30]}...'", False))
            ws_out_queue.put(("audio", pcm48_i16))
            
        except Exception as e:
            print(f"[tts_worker] Error generating audio: {e}")
            ws_out_queue.put(("tts_status", f"❌ Error generating audio: {e}", False))
        
        threading.Timer(0.5, lambda: ws_out_queue.put(("tts_status", "👂 Waiting for text...", False))).start()

# ---------- routes ----------
@app.get("/")
def index():
    return FileResponse("index.html")

# ---------- main ----------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DiaChat - Real-time voice chat with AI")
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip pre-flight checks (faster startup, use when everything is already working)"
    )
    args = parser.parse_args()
    
    # Run pre-flight checks (unless skipped)
    if not args.skip_checks:
        if not run_preflight_checks():
            print("❌ Pre-flight checks failed. Exiting.")
            sys.exit(1)
    else:
        print("⚡ Skipping pre-flight checks for faster startup...")
    
    # Start background threads
    print("[main] Starting background workers...")
    for worker in (stt_worker, llm_worker, tts_worker):
        threading.Thread(target=worker, daemon=True).start()
    
    # Check if SSL certificates exist (decision point after pre-flight attempts)
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"[main] Starting HTTPS server on port 8443...")
        print(f"[main] Access via: https://192.168.194.33:8443")
        print(f"[main] Or locally: https://localhost:8443")
        print(f"[main] Note: You'll need to accept the self-signed certificate warning")
        
        # Run HTTPS server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8443,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file
        )
    else:
        print(f"[main] SSL certificates not found. Starting HTTP server on port 8000...")
        print(f"[main] For microphone access, use: http://localhost:8000")
        print(f"[main] Remote access via IP requires HTTPS!")
        
        # Run HTTP server
        uvicorn.run(app, host="0.0.0.0", port=8000)