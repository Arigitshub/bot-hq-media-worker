# Model and dependency notices

The MIT license covers the original worker code. Model weights are downloaded
from their publishers; they are not included in this source repository. Each
model keeps its own license, attribution, and notices. `models.json` is the
authoritative set of immutable revisions.

| Function | Publisher and model | Pinned revision | Publisher license |
| --- | --- | --- | --- |
| Image | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6) | f332072aa78be7aecdf3ee76d5c247082da564a6 | Apache-2.0 |
| Video | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b) | 0fad780a534b6463e45facd96134c9f345acfa5b | Apache-2.0 |
| Transcription | [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base/tree/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66) | ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66 | MIT |
| Speech | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987) | f3ff3571791e39611d31c381e3a41a3af07b4987 | Apache-2.0 |

These publisher licenses permit commercial use, subject to their terms. They do
not promise that all generated outputs clear copyright, trademark, likeness, or
third-party rights. This worker does not grant rights to user-supplied content.
Training and fine-tuning are separate workloads and are not implemented here.

Software dependencies retain their upstream licenses. Their installed package
metadata and license files remain in the container; `build-manifest.txt` records
the exact resolved Python packages. CUDA libraries retain NVIDIA's applicable
terms; Ubuntu packages retain their distribution notices. The base image is an
unmodified official Runpod image with our additional worker layer. No proprietary
AI service code or credentials are included.
