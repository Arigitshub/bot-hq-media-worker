"""One isolated model job; invoked only by runner.py on the cloud worker.

The parent enforces a wall deadline and kills this process group on expiry.
Models are loaded from pinned official snapshots and are never user-selected.
"""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from policy import AUDIO_TYPES, MODELS, RequestError, artifact_result, failure, validate


def snapshot(kind):
    from huggingface_hub import snapshot_download
    model = MODELS[kind]
    patterns = {
        "image": ["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*"],
        "video": ["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*"],
        "transcribe": ["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json", "README.md"],
        "speech": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "README.md", "LICENSE*"],
    }
    return Path(snapshot_download(model["id"], revision=model["revision"], allow_patterns=patterns[kind],
                                  cache_dir=os.environ.get("HF_HUB_CACHE")))


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RequestError("gpu_required", "Image and video generation require the configured cloud GPU.")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RequestError("gpu_memory", "This image/video worker requires at least 16 GB of GPU memory.")
    return torch


def image(value):
    torch = require_gpu()
    from diffusers import ZImagePipeline
    config = MODELS["image"]
    pipe = ZImagePipeline.from_pretrained(str(snapshot("image")), torch_dtype=torch.bfloat16,
                                         local_files_only=True, use_safetensors=True)
    # Offload components between steps; only this job's pipeline exists in the process.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        result = pipe(value["prompt"], width=config["width"], height=config["height"],
                      num_inference_steps=config["steps"], guidance_scale=0.0,
                      generator=torch.Generator("cuda").manual_seed(value["seed"])).images[0]
    output = io.BytesIO()
    result.save(output, format="PNG")
    return artifact_result("image", output.getvalue(), "image/png",
                           {"width": config["width"], "height": config["height"], "steps": config["steps"], "seed": value["seed"]})


def video(value):
    torch = require_gpu()
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
    from diffusers.utils import export_to_video
    config, local = MODELS["video"], snapshot("video")
    vae = AutoencoderKLWan.from_pretrained(str(local), subfolder="vae", torch_dtype=torch.float32,
                                         local_files_only=True, use_safetensors=True)
    pipe = WanPipeline.from_pretrained(str(local), vae=vae, torch_dtype=torch.bfloat16,
                                      local_files_only=True, use_safetensors=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        frames = pipe(prompt=value["prompt"], width=config["width"], height=config["height"],
                      num_frames=config["frames"], num_inference_steps=config["steps"], guidance_scale=5.0,
                      generator=torch.Generator("cuda").manual_seed(value["seed"])).frames[0]
    with tempfile.TemporaryDirectory(prefix="hq-video-") as directory:
        destination = Path(directory) / "output.mp4"
        export_to_video(frames, str(destination), fps=config["fps"])
        content = destination.read_bytes()
    return artifact_result("video", content, "video/mp4", {
        "width": config["width"], "height": config["height"], "frames": len(frames), "fps": config["fps"],
        "seconds": len(frames) / config["fps"], "steps": config["steps"], "seed": value["seed"],
        "quality": "short low-resolution trial; not a top-model quality benchmark"})


def normalized_audio(value, directory):
    """Decode at most 61 seconds, without accepting playlists, URLs or network protocols."""
    source = Path(directory) / ("input" + AUDIO_TYPES[value["audio_mime"]])
    target = Path(directory) / "mono.wav"
    source.write_bytes(base64.b64decode(value["audio_base64"], validate=True))
    # Fixed argv, no shell and no URL input. Restrict FFmpeg protocols to the local file.
    demuxer = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/webm": "matroska",
               "audio/ogg": "ogg", "audio/mp4": "mov"}[value["audio_mime"]]
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
               "-max_alloc", "134217728", "-threads", "1", "-filter_threads", "1", "-f", demuxer,
               "-i", str(source), "-t", "61", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-y", str(target)]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=20, check=False)
    if result.returncode or not target.is_file():
        raise RequestError("audio_decode", "The audio could not be decoded.")
    import wave
    with wave.open(str(target), "rb") as source_wave:
        seconds = source_wave.getnframes() / source_wave.getframerate()
    if seconds <= 0 or seconds > MODELS["transcribe"]["max_seconds"]:
        raise RequestError("audio_duration", "Transcription supports clips up to 60 seconds.")
    return target, seconds


def read_normalized_pcm(source):
    """Read our own bounded FFmpeg PCM output without invoking a second decoder."""
    import wave
    with wave.open(str(source), "rb") as normalized:
        count = normalized.getnframes()
        if (normalized.getnchannels() != 1 or normalized.getsampwidth() != 2
                or normalized.getframerate() != 16000 or normalized.getcomptype() != "NONE"
                or not 0 < count <= 16000 * MODELS["transcribe"]["max_seconds"]):
            raise RequestError("audio_decode", "The normalized audio has an invalid format.")
        pcm = normalized.readframes(count)
    if len(pcm) != count * 2:
        raise RequestError("audio_decode", "The normalized audio is incomplete.")
    return pcm


def transcribe(value):
    import numpy as np
    from faster_whisper import WhisperModel
    with tempfile.TemporaryDirectory(prefix="hq-audio-") as directory:
        source, seconds = normalized_audio(value, directory)
        # Faster-Whisper accepts a 16 kHz float32 ndarray directly. Reusing our
        # validated PCM avoids PyAV API drift and decoding untrusted media twice.
        samples = np.frombuffer(read_normalized_pcm(source), dtype="<i2").astype(np.float32) / 32768.0
        model = WhisperModel(str(snapshot("transcribe")), device="cpu", compute_type="int8", cpu_threads=2,
                             local_files_only=True)
        iterator, info = model.transcribe(samples, beam_size=1, vad_filter=False,
                                         condition_on_previous_text=False, word_timestamps=False)
        segments, texts = [], []
        for part in iterator:
            if len(segments) >= 200:
                raise RequestError("transcript_limit", "The transcript exceeded its segment limit.")
            text = part.text.strip()
            texts.append(text)
            segments.append({"start": round(part.start, 3), "end": round(part.end, 3), "text": text})
        text = " ".join(texts).strip()
        if len(text) > 20000:
            raise RequestError("transcript_limit", "The transcript exceeded its text limit.")
    config = MODELS["transcribe"]
    return {"ok": True, "kind": "transcribe", "model": config["id"], "revision": config["revision"],
            "text": text, "segments": segments,
            "metadata": {"seconds": round(seconds, 3), "language": info.language, "human_verified": False}}


def speech(value):
    import numpy as np
    import soundfile
    from kokoro import KModel, KPipeline
    config, local = MODELS["speech"], snapshot("speech")
    # Both weights and the voice are explicit pinned local files, not library default-main downloads.
    model = KModel(repo_id=config["id"], config=str(local / "config.json"),
                   model=str(local / "kokoro-v1_0.pth")).to("cpu").eval()
    pipe = KPipeline(lang_code="a", repo_id=config["id"], model=model, device="cpu")
    chunks, samples = [], 0
    for _, _, audio in pipe(value["text"], voice=str(local / "voices/af_heart.pt"), speed=1):
        part = audio.detach().cpu().numpy()
        samples += len(part)
        if samples > config["sample_rate"] * 90:
            raise RequestError("speech_duration", "Speech output exceeded 90 seconds.")
        chunks.append(part)
    if not chunks:
        raise RequestError("empty_speech", "The model produced no audio.")
    output = io.BytesIO()
    soundfile.write(output, np.concatenate(chunks), config["sample_rate"], format="WAV", subtype="PCM_16")
    return artifact_result("speech", output.getvalue(), "audio/wav", {
        "voice": config["voice"], "sample_rate": config["sample_rate"], "seconds": samples / config["sample_rate"],
        "voice_cloning": False})


def generate(value):
    value = validate(value)
    started = time.monotonic()
    result = {"image": image, "video": video, "transcribe": transcribe, "speech": speech}[value["kind"]](value)
    result.setdefault("metadata", {})["runtime_seconds"] = round(time.monotonic() - started, 3)
    return result


if __name__ == "__main__":
    # Paths are generated by the trusted parent, never by request input.
    try:
        if sys.platform == "linux" and os.environ.get("BOT_HQ_PARENT_PID"):
            import ctypes
            import signal
            expected_parent = int(os.environ["BOT_HQ_PARENT_PID"])
            # The child must not continue using a GPU if its controlling worker dies.
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
                raise RuntimeError("Unable to bind child process lifetime")
            if os.getppid() != expected_parent:
                raise RuntimeError("Controlling worker exited")
        result = generate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except RequestError as error:
        result = failure(error.code, str(error))
    except Exception as error:
        # Do not return prompts, transcripts, tokens, paths or upstream exception bodies.
        result = failure("generation_failed", "The model could not complete this request. No automatic retry was made.")
        # Class name only, for an administrator to distinguish missing dependency/OOM failures.
        print("generation_error_class=" + type(error).__name__, file=sys.stderr)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
