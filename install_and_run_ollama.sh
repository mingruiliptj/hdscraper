#!/bin/bash

# Exit on any error
set -e

# Variables
OLLAMA_VERSION="latest"  # Use the latest Ollama version
MODEL_URL="huihui_ai/deepseek-r1-abliterated:14b"
MODEL_DIR="/var/model"
OLLAMA_HOST="0.0.0.0:11434"  # Bind to all interfaces

# Step 1: Update the system
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

# Step 2: Install prerequisites (curl for downloading Ollama)
echo "Installing curl..."
apt-get install -y curl

# Step 3: Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Step 4: Create model directory and set permissions
echo "Setting up model directory at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"
chown root:root "$MODEL_DIR"
chmod 755 "$MODEL_DIR"

# Step 5: Configure Ollama to use /var/model for model storage
echo "Configuring Ollama to use $MODEL_DIR for models..."
export OLLAMA_MODELS="$MODEL_DIR"
echo "export OLLAMA_MODELS=$MODEL_DIR" >> /etc/environment

# Step 6: Pull the DeepSeek R1 Abliterated 14B model
echo "Pulling model $MODEL_URL..."
OLLAMA_MODELS="$MODEL_DIR" ollama pull "$MODEL_URL"

# Step 7: Verify the model is downloaded
echo "Listing installed models..."
ollama list

# Step 8: Run Ollama server in the background
echo "Starting Ollama server on $OLLAMA_HOST..."
export OLLAMA_HOST="$OLLAMA_HOST"
nohup ollama serve > /var/log/ollama.log 2>&1 &

# Step 9: Wait briefly and check if Ollama is running
sleep 5
if curl -s http://localhost:11434/ | grep -q "Ollama is running"; then
    echo "Ollama is running successfully on $OLLAMA_HOST!"
else
    echo "Failed to start Ollama. Check /var/log/ollama.log for details."
    exit 1
fi

# Step 10: Instructions for testing
echo "Ollama is installed and running."
echo "To test the model, run this command in another terminal:"
echo "curl -X POST http://localhost:11434/api/chat -d '{\"model\": \"$MODEL_URL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}]}'"
echo "Or connect remotely using the instance's external IP: http://<EXTERNAL_IP>:11434"