#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 carpaty
set -euo pipefail

# Fix locale warning
export LC_ALL=C
export LANG=C

SESSION="mywork"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMAND="source $HOME/ai-models/.venv/bin/activate && pushd \"$BASE_DIR\" && python3.12 ./nsfw.py --models-path=../models"

# Helper functions
session_exists() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

start_session() {
    echo "🚀 Starting session '$SESSION'..."
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "$COMMAND" C-m
    echo "✅ Session '$SESSION' started successfully"
}

stop_session() {
    if session_exists; then
        echo "🛑 Stopping session '$SESSION'..."
        tmux kill-session -t "$SESSION"
        echo "✅ Session stopped"
    else
        echo "ℹ️ Session '$SESSION' is not running"
    fi
}

attach_session() {
    if session_exists; then
        echo "🔗 Attaching to session '$SESSION'..."
        tmux attach-session -t "$SESSION"
    else
        echo "❌ Error: Session '$SESSION' does not exist"
        echo "Start it with: $0 start"
        exit 1
    fi
}

show_status() {
    if session_exists; then
        echo "✅ Session '$SESSION' is running"
        echo "Attach with: $0 attach"
        exit 0
    else
        echo "⚠️ Session '$SESSION' is not running"
        echo "Start with: $0 start"
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: $0 [command]

Commands:
  start      Start the session (default if no session exists)
  stop       Stop the session
  restart    Stop and start the session
  attach     Attach to existing session
  status     Check if session is running
  help       Show this help message

If no command is given and session doesn't exist: starts and attaches
If no command is given and session exists: shows status
EOF
}

# Main logic
case "${1:-}" in
    start)
        if session_exists; then
            echo "Session '$SESSION' already exists"
            echo "Use '$0 restart' to restart or '$0 attach' to connect"
            exit 1
        fi
        start_session
        ;;

    stop)
        stop_session
        ;;

    restart)
        stop_session
        sleep 1
        start_session
        ;;

    attach)
        attach_session
        ;;

    status)
        show_status
        ;;

    help|--help|-h)
        show_usage
        ;;

    "")
        # No argument: smart default behavior
        if session_exists; then
            show_status
        else
            start_session
            attach_session
        fi
        ;;

    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac
