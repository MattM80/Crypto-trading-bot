#!/bin/zsh

set -euo pipefail

LABEL="${BOT_WATCHDOG_LABEL:-com.lucasaust.crypto-trading-bot}"
SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
DEFAULT_PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
PYTHON_BIN="${BOT_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
LOG_DIR="$PROJECT_ROOT/logs"
STDOUT_LOG="$LOG_DIR/launchd_watchdog.out.log"
STDERR_LOG="$LOG_DIR/launchd_watchdog.err.log"
UID_VALUE="$(id -u)"
DOMAIN="gui/${UID_VALUE}"
LIVE_TRADING="${BOT_ENABLE_LIVE_TRADING:-true}"
SPOT_ONLY="${BOT_SPOT_ONLY_MODE:-true}"
EVIDENCE_MODE_VALUE="${BOT_EVIDENCE_MODE:-true}"
VALIDATION_MODE="${BOT_VALIDATION_ACCOUNT_MODE:-true}"
EXTERNAL_EXPORT="${BOT_ENABLE_EXTERNAL_SIGNAL_EXPORT:-false}"
FUTURES_TRADING="${BOT_ENABLE_FUTURES_TRADING:-false}"

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

write_plist() {
  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>run_final_bot.py</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${PROJECT_ROOT}</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>ENABLE_LIVE_TRADING</key>
    <string>${LIVE_TRADING}</string>
    <key>SPOT_ONLY_MODE</key>
    <string>${SPOT_ONLY}</string>
    <key>EVIDENCE_MODE</key>
    <string>${EVIDENCE_MODE_VALUE}</string>
    <key>VALIDATION_ACCOUNT_MODE</key>
    <string>${VALIDATION_MODE}</string>
    <key>ENABLE_EXTERNAL_SIGNAL_EXPORT</key>
    <string>${EXTERNAL_EXPORT}</string>
    <key>ENABLE_FUTURES_TRADING</key>
    <string>${FUTURES_TRADING}</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>ThrottleInterval</key>
  <integer>15</integer>

  <key>StandardOutPath</key>
  <string>${STDOUT_LOG}</string>

  <key>StandardErrorPath</key>
  <string>${STDERR_LOG}</string>
</dict>
</plist>
EOF
}

bootout_if_loaded() {
  launchctl bootout "$DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true
}

install_job() {
  write_plist
  bootout_if_loaded
  launchctl bootstrap "$DOMAIN" "$PLIST_PATH"
  launchctl kickstart -k "$DOMAIN/$LABEL"
  status_job
}

uninstall_job() {
  bootout_if_loaded
  rm -f "$PLIST_PATH"
  echo "Removed ${LABEL}"
}

status_job() {
  launchctl print "$DOMAIN/$LABEL" | grep -E 'pid =|state =|path =|program ='
}

case "${1:-status}" in
  install)
    install_job
    ;;
  restart)
    install_job
    ;;
  uninstall|remove|stop)
    uninstall_job
    ;;
  status)
    status_job
    ;;
  *)
    echo "Usage: $0 {install|restart|status|uninstall}" >&2
    exit 1
    ;;
esac