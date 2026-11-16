"""Regression coverage for the live PyAV metadata_errors incompatibility."""
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy
import runtime


def write_wav(path, rate=16000, channels=1, pcm=None):
    pcm = pcm if pcm is not None else struct.pack("<hhhh", -32768, -1, 0, 32767)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)
    return pcm


class PcmTests(unittest.TestCase):
    def test_preserves_signed_little_endian_pcm_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "normalized.wav"
            expected = write_wav(source)
            self.assertEqual(runtime.read_normalized_pcm(source), expected)

    def test_rejects_unexpected_rate_and_multichannel_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "normalized.wav"
            for rate, channels in ((24000, 1), (16000, 2)):
                write_wav(source, rate, channels)
                with self.assertRaises(policy.RequestError):
                    runtime.read_normalized_pcm(source)

    def test_passes_samples_not_a_filename_to_whisper(self):
        class Samples:
            def astype(self, dtype):
                self.dtype = dtype
                return self

            def __truediv__(self, divisor):
                self.divisor = divisor
                return self

        samples = Samples()
        numpy = SimpleNamespace(frombuffer=Mock(return_value=samples), float32=object())
        model = SimpleNamespace(transcribe=Mock(return_value=(iter([
            SimpleNamespace(start=0, end=1, text="hello")
        ]), SimpleNamespace(language="en"))))
        whisper = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "normalized.wav"
            pcm = write_wav(source)
            with patch.dict(sys.modules, {"numpy": numpy, "faster_whisper": whisper}), \
                    patch.object(runtime, "normalized_audio", return_value=(source, 1)), \
                    patch.object(runtime, "snapshot", return_value=Path(directory)):
                result = runtime.transcribe({"kind": "transcribe"})
        self.assertEqual(result["text"], "hello")
        numpy.frombuffer.assert_called_once_with(pcm, dtype="<i2")
        self.assertIs(model.transcribe.call_args.args[0], samples)
        self.assertIs(samples.dtype, numpy.float32)
        self.assertEqual(samples.divisor, 32768.0)


if __name__ == "__main__":
    unittest.main()
