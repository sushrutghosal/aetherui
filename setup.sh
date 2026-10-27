#!/usr/bin/env bash
# One-time setup: creates .venv, installs everything, creates .env.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
echo "Using $PY ($($PY --version))"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m playwright install chromium
if [ ! -f .env ]; then
  cp .env.example .env
  read -r -p "Paste your Nebius API key: " KEY
  sed -i.bak "s|^NEBIUS_API_KEY=.*|NEBIUS_API_KEY=$KEY|" .env && rm -f .env.bak
fi
echo
echo "Setup done. Next:"
echo "  ./run.sh       # start AetherUI at http://127.0.0.1:8080"
echo "  ./connect.sh   # connect all compiled sites to Claude Code (one MCP connector)"
