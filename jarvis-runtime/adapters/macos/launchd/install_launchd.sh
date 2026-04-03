#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/com.jarvis.runtime.plist"
LABEL="com.jarvis.runtime"

CURRENT_USER="$(whoami)"
HOME_DIR="$(eval echo ~$CURRENT_USER)"
PLIST_DEST="$HOME_DIR/Library/LaunchAgents/$LABEL.plist"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_PATH="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_PATH="$(which python3)"
else
    echo "Error: No Python found. Activate a virtualenv or install Python 3.11+." >&2
    exit 1
fi

echo "Installing Jarvis launchd agent..."
echo "  User:       $CURRENT_USER"
echo "  Python:     $PYTHON_PATH"
echo "  Project:    $PROJECT_DIR"
echo "  Plist:      $PLIST_DEST"

mkdir -p "$HOME_DIR/.jarvis"
mkdir -p "$(dirname "$PLIST_DEST")"

sed -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    -e "s|__WORKING_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME_DIR|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo "Jarvis daemon installed and started."
echo "  Stop:   launchctl unload $PLIST_DEST"
echo "  Start:  launchctl load $PLIST_DEST"
echo "  Logs:   $HOME_DIR/.jarvis/"
