"""
Qwen3-TTS Apple Silicon API Server
OpenAI-compatible HTTP API wrapper for Qwen3-TTS with queue management and caching
"""

import os
import sys
import io
import shutil
import time
import gc
import warnings
import hashlib
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to path for importing mlx_audio
QWEN_TTS_DIR = Path(__file__).parent
if QWEN_TTS_DIR.exists():
    sys.path.insert(0, str(QWEN_TTS_DIR))

try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError as e:
    print(f"Error: 'mlx_audio' library not found. Please install dependencies first.")
    print(f"Run: pip install -r requirements.txt")
    raise e

# Configuration
DEFAULT_PORT = int(os.getenv("PORT", 8825))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
MODELS_DIR = QWEN_TTS_DIR / "models"
OUTPUT_DIR = QWEN_TTS_DIR / "outputs"
CACHE_DIR = QWEN_TTS_DIR / "cache"
VOICES_DIR = QWEN_TTS_DIR / "voices"
UPLOAD_DIR = QWEN_TTS_DIR / "uploads"

# Create directories
for d in [OUTPUT_DIR, CACHE_DIR, VOICES_DIR, UPLOAD_DIR]:
    d.mkdir(exist_ok=True)

# Model definitions
MODELS = {
    "custom-1.7b": {
        "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "mode": "custom",
        "description": "Custom Voice with emotion control (1.7B)"
    },
    "custom-0.6b": {
        "folder": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "mode": "custom",
        "description": "Custom Voice with emotion control (0.6B - faster)"
    },
    "design-1.7b": {
        "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "mode": "design",
        "description": "Voice Design from text description (1.7B)"
    },
    "design-0.6b": {
        "folder": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
        "mode": "design",
        "description": "Voice Design from text description (0.6B - faster)"
    },
    "clone-1.7b": {
        "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mode": "clone",
        "description": "Voice Cloning from audio (1.7B)"
    },
    "clone-0.6b": {
        "folder": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "mode": "clone",
        "description": "Voice Cloning from audio (0.6B - faster)"
    }
}

# Voice mapping (for OpenAI compatibility)
VOICE_MAP = {
    # OpenAI-compatible voices
    "alloy": "Vivian",
    "echo": "Aiden",
    "fable": "Ethan",
    "onyx": "Ryan",
    "nova": "Serena",
    "shimmer": "Chelsie",
    # Qwen3-TTS native voices
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "ethan": "Ethan",
    "chelsie": "Chelsie",
    # Additional voices for Calligraph-TTS compatibility
    "ono_anna": "Ono_Anna",  # Japanese voice
    "sohee": "Sohee",        # Korean voice
    # Default
    "default": "Vivian"
}

# Speaker map by language
SPEAKER_MAP = {
    "chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "english": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "japanese": ["Ono_Anna"],
    "korean": ["Sohee"]
}

# Model aliases for compatibility
MODEL_ALIASES = {
    "qwen3-tts": "custom-0.6b",
    "tts-1": "custom-0.6b",
    "tts-1-hd": "custom-1.7b",
}

# Global model cache
_model_cache = {}

# Queue management
@dataclass
class QueueItem:
    id: str
    text: str
    voice: str
    model: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    audio_path: Optional[str] = None
    cache_key: Optional[str] = None

class TTSQueue:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.pending: List[QueueItem] = []
        self.processing: List[QueueItem] = []
        self.completed: List[QueueItem] = []
        self.failed: List[QueueItem] = []
        self.lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def add(self, text: str, voice: str, model: str) -> QueueItem:
        cache_key = self._generate_cache_key(text, voice, model)
        item_id = hashlib.md5(f"{cache_key}_{time.time()}".encode()).hexdigest()[:12]

        # Check cache first
        cached_path = self._get_cached_path(cache_key)
        if cached_path.exists():
            item = QueueItem(
                id=item_id,
                text=text,
                voice=voice,
                model=model,
                status="completed",
                completed_at=time.time(),
                audio_path=str(cached_path),
                cache_key=cache_key
            )
            async with self.lock:
                self.completed.append(item)
            return item

        item = QueueItem(
            id=item_id,
            text=text,
            voice=voice,
            model=model,
            cache_key=cache_key
        )
        async with self.lock:
            self.pending.append(item)
        return item

    async def get_next(self) -> Optional[QueueItem]:
        async with self.lock:
            if len(self.processing) >= self.max_concurrent:
                return None
            if not self.pending:
                return None
            item = self.pending.pop(0)
            item.status = "processing"
            item.started_at = time.time()
            self.processing.append(item)
            return item

    async def complete(self, item_id: str, audio_path: str):
        async with self.lock:
            for item in self.processing:
                if item.id == item_id:
                    item.status = "completed"
                    item.completed_at = time.time()
                    item.audio_path = audio_path
                    self.processing.remove(item)
                    self.completed.append(item)
                    # Copy to cache
                    if item.cache_key:
                        cache_path = self._get_cached_path(item.cache_key)
                        shutil.copy(audio_path, cache_path)
                    return

    async def fail(self, item_id: str, error: str):
        async with self.lock:
            for item in self.processing:
                if item.id == item_id:
                    item.status = "failed"
                    item.completed_at = time.time()
                    item.error = error
                    self.processing.remove(item)
                    self.failed.append(item)
                    return

    def _generate_cache_key(self, text: str, voice: str, model: str) -> str:
        content = f"{text}|{voice}|{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_path(self, cache_key: str) -> Path:
        return CACHE_DIR / f"{cache_key}.wav"

    def get_status(self) -> Dict[str, Any]:
        return {
            "pending": len(self.pending),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "is_processing": len(self.processing) > 0
        }

    def get_queue_details(self) -> Dict[str, List[Dict]]:
        return {
            "pending": [self._item_to_dict(i) for i in self.pending],
            "processing": [self._item_to_dict(i) for i in self.processing],
            "completed": [self._item_to_dict(i) for i in self.completed[-50:]],  # Last 50
            "failed": [self._item_to_dict(i) for i in self.failed[-20:]]  # Last 20
        }

    def _item_to_dict(self, item: QueueItem) -> Dict:
        return {
            "id": item.id,
            "text": item.text[:50] + "..." if len(item.text) > 50 else item.text,
            "voice": item.voice,
            "model": item.model,
            "status": item.status,
            "created_at": item.created_at,
            "started_at": item.started_at,
            "completed_at": item.completed_at,
            "error": item.error,
            "audio_path": item.audio_path
        }

    def get_item(self, item_id: str) -> Optional[QueueItem]:
        for queue in [self.pending, self.processing, self.completed, self.failed]:
            for item in queue:
                if item.id == item_id:
                    return item
        return None

# Global queue instance
tts_queue = TTSQueue(max_concurrent=3)

# Background queue processor
async def process_queue():
    """Background task to process TTS queue"""
    while True:
        try:
            item = await tts_queue.get_next()
            if item:
                try:
                    # Generate TTS in thread pool to not block
                    loop = asyncio.get_event_loop()
                    audio_path = await loop.run_in_executor(
                        tts_queue._executor,
                        _generate_tts_sync,
                        item.text,
                        item.voice,
                        item.model
                    )
                    await tts_queue.complete(item.id, audio_path)
                except Exception as e:
                    await tts_queue.fail(item.id, str(e))
            else:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Queue processor error: {e}")
            await asyncio.sleep(1)

def _generate_tts_sync(text: str, voice: str, model_key: str) -> str:
    """Synchronous TTS generation for thread pool"""
    model = load_model_cached(model_key)
    model_info = MODELS[model_key]

    temp_dir = OUTPUT_DIR / f"queue_{int(time.time() * 1000)}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        mode = model_info["mode"]

        if mode == "custom":
            generate_audio(
                model=model,
                text=text,
                voice=voice,
                instruct="Normal tone",
                speed=1.0,
                output_path=str(temp_dir)
            )
        elif mode == "design":
            generate_audio(
                model=model,
                text=text,
                instruct=f"Voice like {voice}",
                output_path=str(temp_dir)
            )
        else:  # clone mode
            generate_audio(
                model=model,
                text=text,
                voice=voice,
                instruct="Normal tone",
                speed=1.0,
                output_path=str(temp_dir)
            )

        audio_files = list(temp_dir.glob("*.wav"))
        if not audio_files:
            raise RuntimeError("No audio file generated")

        # Move to output with meaningful name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = text[:30].replace(" ", "_").replace("/", "_")
        final_name = f"{timestamp}_{safe_text}.wav"
        final_path = OUTPUT_DIR / final_name
        shutil.move(str(audio_files[0]), str(final_path))

        return str(final_path)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        clean_memory()

def get_smart_path(folder_name: str) -> Optional[Path]:
    """Find model path, handling huggingface snapshot folders."""
    full_path = MODELS_DIR / folder_name
    if not full_path.exists():
        return None

    snapshots_dir = full_path / "snapshots"
    if snapshots_dir.exists():
        subfolders = [f for f in snapshots_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
        if subfolders:
            return subfolders[0]

    return full_path

def load_model_cached(model_key: str):
    """Load and cache model to avoid reloading."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model_info = MODELS[model_key]
    model_path = get_smart_path(model_info["folder"])

    if not model_path:
        raise FileNotFoundError(f"Model not found: {model_info['folder']}. Please download it from HuggingFace.")

    print(f"Loading model: {model_key}...")
    model = load_model(str(model_path))
    _model_cache[model_key] = model
    print(f"Model loaded: {model_key}")
    return model

def clean_memory():
    """Force garbage collection."""
    gc.collect()

def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format using ffmpeg"""
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", input_path,
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print(f"Qwen3-TTS API Server starting...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Available models: {list(MODELS.keys())}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Start queue processor
    queue_task = asyncio.create_task(process_queue())

    yield

    # Shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass
    tts_queue._executor.shutdown(wait=True)
    print("Server shutdown.")

app = FastAPI(
    title="Qwen3-TTS API",
    description="OpenAI-compatible API for Qwen3-TTS on Apple Silicon with queue management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SpeechRequest(BaseModel):
    model: str = Field(default="custom-0.6b", description="Model ID to use")
    input: str = Field(..., description="Text to synthesize", max_length=5000)
    voice: str = Field(default="vivian", description="Voice ID")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    instruct: Optional[str] = Field(default=None, description="Emotion instruction (e.g., 'excited and happy')")
    language: Optional[str] = Field(default=None, description="Language hint (chinese, english, japanese, korean)")

class VoiceDesignRequest(BaseModel):
    model: str = Field(default="design-0.6b", description="Model ID to use")
    input: str = Field(..., description="Text to synthesize", max_length=5000)
    voice_description: str = Field(..., description="Description of the desired voice (e.g., 'deep male narrator')")
    response_format: Literal["mp3", "wav"] = Field(default="wav")

class VoiceCloneRequest(BaseModel):
    model: str = Field(default="clone-0.6b", description="Model ID to use")
    input: str = Field(..., description="Text to synthesize", max_length=5000)
    reference_audio: str = Field(..., description="Path to reference audio file")
    reference_text: Optional[str] = Field(default=None, description="Transcript of reference audio")

class QueueStatusResponse(BaseModel):
    pending: int
    processing: int
    completed: int
    failed: int
    is_processing: bool

class Voice(BaseModel):
    voice_id: str
    name: str
    preview_url: Optional[str] = None
    language: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "qwen3-tts"

# Health check
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(_model_cache),
        "queue": tts_queue.get_status()
    }

# OpenAI-compatible models endpoint
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id=model_id,
                created=now,
                owned_by="qwen3-tts"
            ) for model_id in MODELS.keys()
        ]
    }

# OpenAI-compatible voices endpoint
@app.get("/v1/voices")
async def list_voices(language: Optional[str] = None):
    voices = []

    if language and language.lower() in SPEAKER_MAP:
        # Return voices for specific language
        for voice_id in SPEAKER_MAP[language.lower()]:
            voices.append(Voice(
                voice_id=voice_id.lower(),
                name=voice_id,
                language=language.lower()
            ))
    else:
        # Return all voices
        for voice_id, name in VOICE_MAP.items():
            if voice_id == "default":
                continue
            # Determine language
            lang = "english"
            for l, speakers in SPEAKER_MAP.items():
                if name in speakers:
                    lang = l
                    break
            voices.append(Voice(voice_id=voice_id, name=name, language=lang))

    return {"object": "list", "data": voices}

# OpenAI-compatible TTS endpoint
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Create speech from text (OpenAI-compatible endpoint) with queue support."""

    start_time = time.time()

    try:
        # Resolve model alias
        resolved_model = MODEL_ALIASES.get(request.model, request.model)

        # Validate model
        if resolved_model not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}. Available: {list(MODELS.keys())}")

        request.model = resolved_model

        # Resolve voice
        voice_name = VOICE_MAP.get(request.voice.lower(), request.voice)

        # Add to queue
        item = await tts_queue.add(request.input, voice_name, request.model)

        # If already cached, return immediately
        if item.status == "completed" and item.audio_path:
            return FileResponse(
                item.audio_path,
                media_type="audio/wav",
                headers={"X-Cache": "HIT"}
            )

        # Wait for processing (with timeout)
        timeout = 120  # 2 minutes
        waited = 0
        while item.status == "pending" or item.status == "processing":
            await asyncio.sleep(0.1)
            waited += 0.1
            if waited > timeout:
                raise HTTPException(status_code=504, detail="TTS generation timeout")

        if item.status == "failed":
            raise HTTPException(status_code=500, detail=item.error or "TTS generation failed")

        if not item.audio_path or not os.path.exists(item.audio_path):
            raise HTTPException(status_code=500, detail="Audio file not found")

        elapsed = time.time() - start_time
        print(f"Generated audio for '{request.input[:50]}...' in {elapsed:.2f}s")

        return FileResponse(
            item.audio_path,
            media_type="audio/wav",
            headers={"X-Cache": "MISS"}
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Queue management endpoints
@app.get("/v1/queue/status")
async def queue_status() -> QueueStatusResponse:
    """Get TTS queue status"""
    status = tts_queue.get_status()
    return QueueStatusResponse(**status)

@app.get("/v1/queue/details")
async def queue_details():
    """Get detailed queue information"""
    return tts_queue.get_queue_details()

@app.get("/v1/queue/item/{item_id}")
async def queue_item(item_id: str):
    """Get specific queue item status"""
    item = tts_queue.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    return tts_queue._item_to_dict(item)

@app.post("/v1/queue/clear")
async def queue_clear():
    """Clear completed and failed items from queue"""
    async with tts_queue.lock:
        tts_queue.completed.clear()
        tts_queue.failed.clear()
    return {"success": True, "message": "Queue cleared"}

# Voice Design endpoint
@app.post("/v1/audio/design")
async def create_voice_design(request: VoiceDesignRequest):
    """Generate speech with custom voice design"""
    start_time = time.time()

    try:
        model_key = request.model
        if model_key not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_key}")

        model_info = MODELS[model_key]
        if model_info["mode"] != "design":
            raise HTTPException(status_code=400, detail=f"Model {model_key} is not a design model")

        model = load_model_cached(model_key)

        temp_dir = OUTPUT_DIR / f"design_{int(time.time() * 1000)}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            generate_audio(
                model=model,
                text=request.input,
                instruct=request.voice_description,
                output_path=str(temp_dir)
            )

            audio_files = list(temp_dir.glob("*.wav"))
            if not audio_files:
                raise RuntimeError("No audio file generated")

            audio_path = audio_files[0]

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            shutil.rmtree(temp_dir, ignore_errors=True)

            elapsed = time.time() - start_time
            print(f"Generated designed voice audio in {elapsed:.2f}s")

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=designed_speech.wav"}
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            clean_memory()

    except Exception as e:
        print(f"Error in voice design: {e}")
        raise HTTPException(status_code=500, detail=f"Voice design failed: {str(e)}")

# Voice Clone endpoint with file upload
@app.post("/v1/audio/clone")
async def create_voice_clone(
    text: str = Form(..., description="Text to synthesize"),
    reference_text: Optional[str] = Form(None, description="Transcript of reference audio"),
    model: str = Form(default="clone-0.6b", description="Model ID to use"),
    reference_audio: UploadFile = File(..., description="Reference audio file (wav, mp3, etc.)")
):
    """Clone voice from reference audio and generate speech"""
    start_time = time.time()

    try:
        if model not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

        model_info = MODELS[model]
        if model_info["mode"] != "clone":
            raise HTTPException(status_code=400, detail=f"Model {model} is not a clone model")

        # Save uploaded file
        upload_id = f"upload_{int(time.time() * 1000)}"
        upload_path = UPLOAD_DIR / f"{upload_id}_{reference_audio.filename}"
        wav_path = UPLOAD_DIR / f"{upload_id}.wav"

        with open(upload_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        # Convert to WAV if needed
        if not convert_audio_to_wav(str(upload_path), str(wav_path)):
            # If conversion fails, try using original
            wav_path = upload_path

        # Load model and generate
        model_obj = load_model_cached(model)

        temp_dir = OUTPUT_DIR / f"clone_{int(time.time() * 1000)}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            generate_audio(
                model=model_obj,
                text=text,
                ref_audio=str(wav_path),
                ref_text=reference_text or ".",
                output_path=str(temp_dir)
            )

            audio_files = list(temp_dir.glob("*.wav"))
            if not audio_files:
                raise RuntimeError("No audio file generated")

            audio_path = audio_files[0]

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            shutil.rmtree(temp_dir, ignore_errors=True)

            elapsed = time.time() - start_time
            print(f"Generated cloned voice audio in {elapsed:.2f}s")

            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=cloned_speech.wav"}
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if upload_path.exists():
                upload_path.unlink()
            if wav_path.exists() and wav_path != upload_path:
                wav_path.unlink()
            clean_memory()

    except Exception as e:
        print(f"Error in voice clone: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

# Voice management endpoints
@app.get("/v1/voices/saved")
async def list_saved_voices():
    """List saved voice clones"""
    voices = []
    if VOICES_DIR.exists():
        for f in VOICES_DIR.glob("*.wav"):
            name = f.stem
            txt_file = f.with_suffix(".txt")
            has_transcript = txt_file.exists()
            voices.append({
                "name": name,
                "audio_path": str(f),
                "has_transcript": has_transcript,
                "created": os.path.getctime(f)
            })
    return {"voices": voices}

@app.post("/v1/voices/save")
async def save_voice(
    name: str = Form(..., description="Name for the voice"),
    transcript: Optional[str] = Form(None, description="Transcript of the audio"),
    audio: UploadFile = File(..., description="Audio file")
):
    """Save a voice for later use"""
    try:
        # Clean name
        safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid voice name")

        # Save files
        wav_path = VOICES_DIR / f"{safe_name}.wav"
        txt_path = VOICES_DIR / f"{safe_name}.txt"

        upload_path = UPLOAD_DIR / f"temp_{audio.filename}"
        with open(upload_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Convert to WAV
        if not convert_audio_to_wav(str(upload_path), str(wav_path)):
            shutil.copy(upload_path, wav_path)

        # Save transcript
        if transcript:
            with open(txt_path, "w") as f:
                f.write(transcript)

        upload_path.unlink(missing_ok=True)

        return {"success": True, "name": safe_name, "path": str(wav_path)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/voices/saved/{voice_name}")
async def delete_saved_voice(voice_name: str):
    """Delete a saved voice"""
    wav_path = VOICES_DIR / f"{voice_name}.wav"
    txt_path = VOICES_DIR / f"{voice_name}.txt"

    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")

    wav_path.unlink(missing_ok=True)
    txt_path.unlink(missing_ok=True)

    return {"success": True, "message": f"Voice {voice_name} deleted"}

# Cache management
@app.get("/v1/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not CACHE_DIR.exists():
        return {"size": 0, "files": 0}

    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.wav"))
    file_count = len(list(CACHE_DIR.glob("*.wav")))

    return {
        "size": total_size,
        "size_mb": round(total_size / 1024 / 1024, 2),
        "files": file_count
    }

@app.post("/v1/cache/clear")
async def cache_clear():
    """Clear all cached files"""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.wav"):
            f.unlink()
    return {"success": True, "message": "Cache cleared"}

# Audio files management
@app.get("/v1/audio/files")
async def list_audio_files(limit: int = 100):
    """List generated audio files"""
    files = []
    if OUTPUT_DIR.exists():
        all_files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
        for f in all_files[:limit]:
            stat = f.stat()
            files.append({
                "filename": f.name,
                "path": str(f),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return {"files": files}

@app.get("/v1/audio/files/{filename}")
async def get_audio_file(filename: str):
    """Download a specific audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.delete("/v1/audio/files/{filename}")
async def delete_audio_file(filename: str):
    """Delete a specific audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"success": True, "message": f"File {filename} deleted"}

# Simple TTS endpoint with query params (for easier testing)
@app.get("/tts")
async def tts_get(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(default="vivian", description="Voice ID"),
    model: str = Query(default="custom-0.6b", description="Model ID"),
    speed: float = Query(default=1.0, ge=0.5, le=2.0),
    language: Optional[str] = Query(default=None, description="Language hint (Chinese, English, etc.)")
):
    """Simple GET endpoint for TTS (for testing)."""
    request = SpeechRequest(
        model=model,
        input=text,
        voice=voice,
        speed=speed,
        language=language
    )
    return await create_speech(request)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Qwen3-TTS API Server",
        "description": "OpenAI-compatible API for Qwen3-TTS on Apple Silicon with queue management",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "voices": "/v1/voices",
            "speech": "/v1/audio/speech (POST)",
            "design": "/v1/audio/design (POST)",
            "clone": "/v1/audio/clone (POST)",
            "queue": {
                "status": "/v1/queue/status",
                "details": "/v1/queue/details",
                "clear": "/v1/queue/clear (POST)"
            },
            "voices_management": {
                "list": "/v1/voices/saved",
                "save": "/v1/voices/save (POST)",
                "delete": "/v1/voices/saved/{name} (DELETE)"
            },
            "cache": {
                "stats": "/v1/cache/stats",
                "clear": "/v1/cache/clear (POST)"
            },
            "audio_files": {
                "list": "/v1/audio/files",
                "download": "/v1/audio/files/{filename}",
                "delete": "/v1/audio/files/{filename} (DELETE)"
            },
            "tts_get": "/tts (GET)"
        },
        "docs": "/docs"
    }

def main():
    port = int(os.getenv("PORT", DEFAULT_PORT))
    host = os.getenv("HOST", DEFAULT_HOST)

    print(f"Starting Qwen3-TTS API Server on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
