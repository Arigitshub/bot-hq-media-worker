# Bot HQ media worker

A bounded cloud worker for four downloadable models: Z-Image-Turbo images,
Wan 2.1 short videos, Kokoro speech, and Faster-Whisper transcription.
This repository contains only the inference worker. It contains no application
accounts, private files, provider keys, billing data, or user conversations.

The first A40 trial produced a valid 512 px image, a 2.0625-second 512x288 MP4,
a 24 kHz speech WAV, and the expected transcript of a known spoken phrase.
Those tests do not establish production reliability or
quality parity with leading commercial services. The deployment image must pass
its own smoke tests before public requests are enabled.

## Interface

Runpod queue input is `{"input":{"kind":"image","prompt":"A red sailboat","seed":42}}`.
Supported kinds are `image`, `video`, `speech` (with `text`), and `transcribe`
(with plain `audio_base64` and an approved `audio_mime`). Artifact results contain
`artifact_base64`, `mime`, `bytes`, and `sha256`. Transcription returns text and
segments. Errors use `ok:false` plus a safe code and message.

The worker fixes model IDs, revisions, generation settings, output sizes, and a
540-second child-process deadline. It accepts no user code, paths, model URLs,
shell commands, custom workflows, resolution, inference steps, or voice cloning.
Audio input/output and media artifacts are bounded to 6 MiB. Audio input is at
most 60 seconds. Only one job executes per process. A timeout stops inference;
it is not a provider spending cap.

## Cloud deployment

The manual GitHub Actions workflow builds on a free standard runner in a public
repository and publishes a digest-pinned GHCR image. It uses GitHub's generated
workflow token; a personal token with package-write scope is not required.
New GHCR packages default to private: the operator must make the reviewed
worker package publicly pullable or configure a durable read-only credential.
The workflow never deploys a GPU or creates cloud spending by itself.

Use Runpod QUEUE workers, minimum zero and maximum one, with a short idle
timeout. For an image-only or video-only endpoint set `BOT_HQ_KIND` accordingly
and `BOT_HQ_REQUIRE_HOST_CACHE=1`, attach the exact HF revision in `models.json`,
and keep the normal writable `HF_HUB_CACHE=/cache/huggingface/hub`. The wrapper
links the approved host-cached snapshot into that cache without copying weights.
A missing cached revision returns an error instead of downloading a replacement.
Each endpoint supports one Runpod cached model; serialize requests in your app
if several endpoints share one spending budget.

Kokoro and Whisper run on CPU. `install-voice-cloud.sh` installs a loopback-only,
authenticated service on an existing Ubuntu cloud machine. It must not be run
on a user's personal PC. It does not create a cloud machine or purchase credits.
The Supervisor program runs under its own unprivileged UID. Each disposable
model process has a 4-GiB address-space limit, 120/125-second CPU-time limit,
one CPU affinity and 300-second wall deadline. These are workload limits, not
a subscription or billing cap. Test actual RAM and speech latency on the host.

Dependencies are pinned directly; transitive versions are recorded at build.
Run offline tests with `python3 -m unittest discover -s tests -v`. Model inference
and GPU tests are deliberately excluded from this test suite. See
`THIRD_PARTY_NOTICES.md` and `LICENSE` before redistributing.
