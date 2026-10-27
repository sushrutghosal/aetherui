#!/usr/bin/env bash
# Start the AetherUI web app (compile sites, chat, see proof).
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
echo "Open http://127.0.0.1:8080"
exec .venv/bin/python -m uvicorn demo_ui:app --port 8080
