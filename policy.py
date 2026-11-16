"""Dependency-free input/output limits. Importing this module starts no model."""
import base64
import binascii
import hashlib
import json
from pathlib import Path

MAX_AUDIO_BYTES = 6 * 1024 * 1024
MAX_ARTIFACT_BYTES = 6 * 1024 * 1024
MAX_JSON_BYTES = 9 * 1024 * 1024
MAX_SECONDS = 540
AUDIO_TYPES = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a"}
MODELS = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))


class RequestError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate(value):
    if not isinstance(value, dict):
        raise RequestError("invalid_input", "A JSON object is required.")
    kind = value.get("kind")
    allowed = {
        "image": {"kind", "prompt", "seed"},
        "video": {"kind", "prompt", "seed"},
        "speech": {"kind", "text"},
        "transcribe": {"kind", "audio_base64", "audio_mime"},
    }
    if not isinstance(kind, str) or kind not in allowed:
        raise RequestError("unsupported_kind", "Choose image, video, speech or transcribe.")
    if set(value) - allowed[kind]:
        raise RequestError("unsupported_fields", "This worker accepts only its fixed generation options.")
    result = {"kind": kind}
    if kind in ("image", "video"):
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1200:
            raise RequestError("prompt_limit", "Use a prompt of 1 to 1200 characters.")
        seed = value.get("seed", 42)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise RequestError("seed_limit", "Seed must be an integer from 0 to 4294967295.")
        result.update(prompt=prompt.strip(), seed=seed)
    elif kind == "speech":
        text = value.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 600:
            raise RequestError("text_limit", "Speech supports 1 to 600 characters per request.")
        result["text"] = text.strip()
    else:
        mime, encoded = value.get("audio_mime"), value.get("audio_base64")
        if not isinstance(mime, str) or mime not in AUDIO_TYPES:
            raise RequestError("audio_type", "Use WAV, MP3, WebM, Ogg or M4A audio.")
        if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4 * ((MAX_AUDIO_BYTES + 2) // 3):
            raise RequestError("audio_size", "Audio must be no larger than 6 MiB.")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise RequestError("audio_encoding", "Audio must use plain valid base64.") from None
        if not 1 <= len(audio) <= MAX_AUDIO_BYTES:
            raise RequestError("audio_size", "Audio must be no larger than 6 MiB.")
        result.update(audio_base64=encoded, audio_mime=mime)
    return result


def artifact_result(kind, content, mime, metadata=None):
    expected = {"image": "image/png", "video": "video/mp4", "speech": "audio/wav"}
    if kind not in expected or mime != expected[kind]:
        raise RequestError("artifact_type", "The generated artifact has an unexpected format.")
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_ARTIFACT_BYTES:
        raise RequestError("artifact_size", "The generated artifact exceeds the trial size limit.")
    model = MODELS[kind]
    return {"ok": True, "kind": kind, "model": model["id"], "revision": model["revision"],
            "mime": mime, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "artifact_base64": base64.b64encode(content).decode("ascii"), "metadata": metadata or {}}


def failure(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}
