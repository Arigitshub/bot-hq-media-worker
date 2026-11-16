#!/usr/bin/env bash
# Run only on the existing Ubuntu cloud machine, never a personal computer.
# No cloud resources, external accounts, subscriptions, or credits are created.
set -euo pipefail
test "${BOT_HQ_CLOUD_INSTALL:-}" = 1 || { echo 'Set the cloud-only install guard on the intended cloud machine.'; exit 1; }
test "$(uname -s)" = Linux
test "$(id -u)" = 0 || { echo 'Cloud administrator privileges are required.'; exit 1; }
test -d /run/systemd/system || { echo 'This installer requires systemd on the cloud machine.'; exit 1; }
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
install_dir=/opt/bot-hq-voice
state_dir=/var/lib/bot-hq-voice
for file in runtime.py policy.py runner.py http_server.py models.json requirements-cpu.txt; do
  test -f "$source_dir/$file" && test ! -L "$source_dir/$file"
done
if systemctl is-active --quiet bot-hq-voice.service; then
  echo 'Existing voice service is running; review an update before replacing it.'
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv ffmpeg espeak-ng libsndfile1 libgomp1 ca-certificates
id bot-hq-voice >/dev/null 2>&1 || useradd --system --home-dir "$state_dir" --shell /usr/sbin/nologin bot-hq-voice
install -d -m 0755 "$install_dir"
install -d -o bot-hq-voice -g bot-hq-voice -m 0700 "$state_dir" "$state_dir/cache" "$state_dir/cache/hub"
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/python" -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$install_dir/venv/bin/python" -m pip install --no-cache-dir -r "$source_dir/requirements-cpu.txt"
"$install_dir/venv/bin/python" -m pip check
"$install_dir/venv/bin/python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.9.1"; assert torch.version.cuda is None'
for file in runtime.py policy.py runner.py http_server.py models.json; do
  install -m 0644 "$source_dir/$file" "$install_dir/$file"
done
"$install_dir/venv/bin/python" -m pip freeze > "$install_dir/build-manifest.txt"
if test ! -e "$state_dir/worker.token"; then
  umask 077
  "$install_dir/venv/bin/python" -c 'import secrets; from pathlib import Path; Path("/var/lib/bot-hq-voice/worker.token").write_text(secrets.token_urlsafe(48))'
fi
test -f "$state_dir/worker.token" && test ! -L "$state_dir/worker.token"
chown bot-hq-voice:bot-hq-voice "$state_dir/worker.token"
chmod 0600 "$state_dir/worker.token"
cat > /etc/systemd/system/bot-hq-voice.service <<'SERVICE'
[Unit]
Description=Bot HQ private CPU speech and transcription worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bot-hq-voice
Group=bot-hq-voice
WorkingDirectory=/opt/bot-hq-voice
Environment=HOME=/var/lib/bot-hq-voice
Environment=PATH=/opt/bot-hq-voice/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=HF_HOME=/var/lib/bot-hq-voice/cache
Environment=HF_HUB_CACHE=/var/lib/bot-hq-voice/cache/hub
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=OMP_NUM_THREADS=2
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=/opt/bot-hq-voice/venv/bin/python /opt/bot-hq-voice/http_server.py --host 127.0.0.1 --port 8766 --token-file /var/lib/bot-hq-voice/worker.token --kinds speech,transcribe
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
KillMode=control-group
MemoryHigh=2G
MemoryMax=3G
CPUQuota=150%
TasksMax=128
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/bot-hq-voice

[Install]
WantedBy=multi-user.target
SERVICE
systemctl daemon-reload
systemctl enable --now bot-hq-voice.service
systemctl is-active --quiet bot-hq-voice.service
"$install_dir/venv/bin/python" - <<'PY'
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
key = Path('/var/lib/bot-hq-voice/worker.token').read_text().strip()
request = Request('http://127.0.0.1:8766/health', headers={'Authorization': 'Bearer ' + key})
for attempt in range(20):
    try:
        with urlopen(request, timeout=5) as response:
            health = json.load(response)
        break
    except OSError:
        if attempt == 19:
            raise
        time.sleep(0.5)
assert health.get('ok') is True
print('Voice service is listening privately on 127.0.0.1:8766; inference still needs a real test.')
print('The private token file is /var/lib/bot-hq-voice/worker.token; its content was not printed.')
PY
