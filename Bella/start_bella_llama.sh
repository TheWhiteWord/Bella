#!/bin/bash

# Configuration
MODEL_DIR="/media/theww/AI/models/gguf/"
MODEL_CONFIG="./models.ini"
PORT=8080
HOST="0.0.0.0"
# Set this to the absolute path of your llama-server if it's not in PATH
LLAMA_SERVER_BIN="/media/theww/AI/Code/AI/Llamaccp/llama.cpp/build/bin/llama-server"

# Kill any existing llama-server instances to avoid conflicts
pkill llama-server

echo "Starting Bella Llama.cpp Server (Router Mode)..."

# Find llama-server
if [ -n "$LLAMA_SERVER_BIN" ]; then
    SERVER_CMD="$LLAMA_SERVER_BIN"
elif command -v llama-server &> /dev/null; then
    SERVER_CMD="llama-server"
# Common fallback locations
elif [ -f "/usr/local/bin/llama-server" ]; then
    SERVER_CMD="/usr/local/bin/llama-server"
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    SERVER_CMD="$HOME/llama.cpp/build/bin/llama-server"
elif [ -f "$HOME/ai/llama.cpp/build/bin/llama-server" ]; then
    SERVER_CMD="$HOME/ai/llama.cpp/build/bin/llama-server"
else
    echo "Error: llama-server not found in PATH or common locations."
    echo "Please edit this script ($0) and set LLAMA_SERVER_BIN to the path of your llama-server executable."
    exit 1
fi

echo "Using server binary: $SERVER_CMD"
echo "Models Directory: $MODEL_DIR"

# Start the server in the background
"$SERVER_CMD" \
    --models-dir "$MODEL_DIR" \
    --models-preset "$MODEL_CONFIG" \
    --port $PORT \
    --host $HOST \
    --models-max 1 \
    --parallel 1 \
    --cont-batching \
    --n-gpu-layers 99 \
    --ctx-size 4096 \
    > llama_server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID $SERVER_PID. Logs in llama_server.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "http://localhost:$PORT/health" | grep -q "ok"; then
        echo "Server is ready!"
        break
    fi
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server failed to start. Check logs."
        exit 1
    fi
    sleep 1
    echo -n "."
done

if [ $i -eq $MAX_RETRIES ]; then
    echo "Timed out waiting for server."
    kill $SERVER_PID
    exit 1
fi

# Start Bella
echo "Starting Bella..."
python3 main.py "$@"

# Cleanup on exit
echo "Stopping server..."
kill $SERVER_PID
