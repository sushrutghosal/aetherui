#!/usr/bin/env bash
# Register ONE MCP connector ("aether") in Claude Code that serves every compiled site in manifests/.
cd "$(dirname "$0")"
DIR="$(pwd)"
command -v claude >/dev/null || { echo "Claude Code CLI not found. Install it, then re-run."; exit 1; }
claude mcp remove aether >/dev/null 2>&1 || true
claude mcp add --transport stdio aether -- "$DIR/.venv/bin/python" "$DIR/vision_to_mcp.py" serve "$DIR/manifests"
claude mcp list | grep aether
echo "Done. Open a NEW Claude Code session and type /mcp to see the tools."
