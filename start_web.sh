#!/bin/bash

# Startup script for STAT web interface + Jupyter Lab

FLASK_PORT=${1:-8889}
JUPYTER_PORT=${2:-8890}

# Ensure HTTPS proxy is set (needed for POE/external LLM APIs)
export https_proxy="${https_proxy:-${http_proxy:-}}"

echo "========================================"
echo "STAT - Spatial Transcriptomics Analysis"
echo "========================================"
echo ""
echo "Starting services:"
echo "  Web UI:       http://0.0.0.0:${FLASK_PORT}"
echo "  Jupyter Lab:  http://0.0.0.0:${JUPYTER_PORT}"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================"
echo ""

# cd /import/home3/yhchenmath/Code/STAT-agent

# Start Jupyter Lab in background (no token, no browser, allow iframe embedding)
jupyter lab \
    --port=${JUPYTER_PORT} \
    --ip=0.0.0.0 \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy":"frame-ancestors *","X-Frame-Options":""}}' \
    --ServerApp.allow_origin='*' \
    &
JUPYTER_PID=$!

# Cleanup on exit
trap "kill $JUPYTER_PID 2>/dev/null; exit" INT TERM EXIT

# Start Flask app (foreground)
python3 -m stat_agent.web.app --host 0.0.0.0 --port ${FLASK_PORT} --jupyter-port ${JUPYTER_PORT}
