#!/usr/bin/env bash
# Run only on the existing Ubuntu cloud machine, never a personal computer.
# No cloud resources, external accounts, subscriptions, or credits are created.
set -euo pipefail
test "${BOT_HQ_CLOUD_INSTALL:-}" = 1 || { echo 'Set the cloud-only install guard on the intended cloud machine.'; exit 1; }
test "$(uname -s)" = Linux
test "$(id -u)" = 0 || { echo 'Cloud administrator privileges are required.'; exit 1; }
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
install_dir=/opt/bot-hq-voice
state_dir=/var/lib/bot-hq-voice
for file in runtime.py policy.py runner.py http_server.py voice_server.py models.json requirements-cpu.txt; do
  test -f "$source_dir/$file" && test ! -L "$source_dir/$file"
done
if command -v supervisorctl >/dev/null && supervisorctl status bot-hq-voice 2>/dev/null | grep -q RUNNING; then
  echo 'Existing voice service is running; review an update before replacing it.'
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv ffmpeg espeak-ng libsndfile1 libgomp1 ca-certificates supervisor util-linux
supervisorctl pid >/dev/null || { echo 'The existing Supervisor must be running before installing this program.'; exit 1; }
id bot-hq-voice >/dev/null 2>&1 || useradd --system --home-dir "$state_dir" --shell /usr/sbin/nologin bot-hq-voice
install -d -m 0755 "$install_dir"
install -d -o bot-hq-voice -g bot-hq-voice -m 0700 "$state_dir" "$state_dir/cache" "$state_dir/cache/hub"
python3 -m venv "$install_dir/venv"
"$install_dir/venv/bin/python" -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$install_dir/venv/bin/python" -m pip install --no-cache-dir -r "$source_dir/requirements-cpu.txt"
"$install_dir/venv/bin/python" -m pip check
"$install_dir/venv/bin/python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.9.1"; assert torch.version.cuda is None'
"$install_dir/venv/bin/python" -c 'import kokoro, faster_whisper; print("Voice dependency imports verified; no model was loaded.")'
for file in runtime.py policy.py runner.py http_server.py voice_server.py models.json; do
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
install -d -m 0755 /etc/supervisor/conf.d
cat > /etc/supervisor/conf.d/bot-hq-voice.conf <<'SERVICE'
[program:bot-hq-voice]
directory=/opt/bot-hq-voice
command=/opt/bot-hq-voice/venv/bin/python /opt/bot-hq-voice/voice_server.py --token-file /var/lib/bot-hq-voice/worker.token
user=bot-hq-voice
environment=HOME="/var/lib/bot-hq-voice",PATH="/opt/bot-hq-voice/venv/bin:/usr/local/bin:/usr/bin:/bin",HF_HOME="/var/lib/bot-hq-voice/cache",HF_HUB_CACHE="/var/lib/bot-hq-voice/cache/hub",HF_HUB_DISABLE_TELEMETRY="1",OMP_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false"
autostart=true
autorestart=unexpected
startsecs=3
startretries=2
stopwaitsecs=10
stopasgroup=true
killasgroup=true
umask=0077
redirect_stderr=true
stdout_logfile=/var/lib/bot-hq-voice/service.log
stdout_logfile_maxbytes=1MB
stdout_logfile_backups=1
SERVICE
# Targeted update: never restart or reload other applications.
supervisorctl reread
supervisorctl update bot-hq-voice
supervisorctl status bot-hq-voice
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
