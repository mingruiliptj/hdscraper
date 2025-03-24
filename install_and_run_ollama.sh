#!/bin/bash

# Exit on any error
set -e

# Variables
OLLAMA_VERSION="latest"  # Use the latest Ollama version
MODEL_URL="huihui_ai/deepseek-r1-abliterated:14b"
MODEL_DIR="/var/model"
OLLAMA_HOST="0.0.0.0:11434"  # Bind to all interfaces

# Step 1: Check if Ollama is already installed
echo "Checking if Ollama is already installed..."
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama is already installed. Checking version..."
    ollama --version
    echo "Skipping Ollama installation..."
else
    # Step 2: Update the system
    echo "Updating system packages..."
    apt-get update

    # Step 3: Install prerequisites (curl for downloading Ollama)
    echo "Installing curl..."
    apt-get install -y curl

    # Step 4: Install Ollama
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Step 5: Create model directory and set permissions
echo "Setting up model directory at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"
chown root:root "$MODEL_DIR"
chmod 755 "$MODEL_DIR"

# Step 6: Configure Ollama to use /var/model for model storage
echo "Configuring Ollama to use $MODEL_DIR for models..."
export OLLAMA_MODELS="$MODEL_DIR"
echo "export OLLAMA_MODELS=$MODEL_DIR" >> /etc/environment

# Step 7: Start Ollama server first
echo "Starting Ollama server on $OLLAMA_HOST..."
export OLLAMA_HOST="$OLLAMA_HOST"
nohup ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama server to be ready
echo "Waiting for Ollama server to be ready..."
max_retries=30
count=0
while ! curl -s http://localhost:11434/ > /dev/null; do
    sleep 2
    count=$((count + 1))
    if [ $count -ge $max_retries ]; then
        echo "Timeout waiting for Ollama server to start"
        exit 1
    fi
    echo "Waiting for Ollama server... (attempt $count/$max_retries)"
done

# Step 8: Pull the DeepSeek R1 Abliterated 14B model with retry mechanism
echo "Pulling model $MODEL_URL..."
max_pull_retries=3
pull_attempt=1
pull_success=false

while [ $pull_attempt -le $max_pull_retries ] && [ "$pull_success" = false ]; do
    echo "Attempt $pull_attempt to pull model..."
    if OLLAMA_MODELS="$MODEL_DIR" ollama pull "$MODEL_URL"; then
        pull_success=true
        echo "Model pulled successfully!"
    else
        echo "Pull attempt $pull_attempt failed"
        if [ $pull_attempt -lt $max_pull_retries ]; then
            echo "Waiting before retry..."
            sleep 10
        fi
        pull_attempt=$((pull_attempt + 1))
    fi
done

if [ "$pull_success" = false ]; then
    echo "Failed to pull model after $max_pull_retries attempts"
    exit 1
fi

# Step 9: Verify the model is downloaded
echo "Listing installed models..."
if ! ollama list; then
    echo "Failed to list models. Check if Ollama is running properly."
    exit 1
fi

# Step 10: Final verification
if curl -s http://localhost:11434/ | grep -q "Ollama is running"; then
    echo "Ollama is running successfully on $OLLAMA_HOST!"
else
    echo "Failed to verify Ollama is running. Check /var/log/ollama.log for details."
    exit 1
fi

# Step 11: Instructions for testing
echo "Ollama is installed and running."
echo "To test the model, run this command in another terminal:"
echo "curl -X POST http://localhost:11434/api/chat -d '{\"model\": \"$MODEL_URL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}]}'"
echo "Or connect remotely using the instance's external IP: http://<EXTERNAL_IP>:11434"



OLLAMA_HOST=0.0.0.0:11434 OLLAMA_ORIGINS="*"  ollama run huihui_ai/deepseek-r1-abliterated:14b