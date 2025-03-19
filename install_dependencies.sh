#!/bin/bash
# Helper script to install dependencies for the Human Image Scraper in Google Cloud Shell

echo "======== Human Image Scraper Dependency Installer ========"
echo "Detecting Python version..."

PY_VERSION=$(python --version 2>&1)
echo "Found $PY_VERSION"

# Extract major and minor version
PY_MAJOR=$(echo $PY_VERSION | cut -d' ' -f2 | cut -d'.' -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d' ' -f2 | cut -d'.' -f2)

# Handle different Python versions
if [ "$PY_MAJOR" = "3" ] && [ "$PY_MINOR" -ge "13" ]; then
    echo "Python 3.13+ detected - MediaPipe will not be available"
    echo "Installing compatible dependencies..."
    pip install --user numpy>=2.1.0 opencv-python-headless>=4.8.0 requests pillow tqdm duckduckgo-search>=3.9.6
    RESULT=$?

elif [ "$PY_MAJOR" = "3" ] && [ "$PY_MINOR" = "12" ]; then
    echo "Python 3.12 detected - using specialized requirements"
    echo "Installing pre-built NumPy first..."
    pip install --user --only-binary=:all: numpy==1.26.4
    
    echo "Installing remaining dependencies..."
    if [ -f "requirements_cloud_py312.txt" ]; then
        pip install --user -r requirements_cloud_py312.txt
    else
        pip install --user --only-binary=:all: opencv-python-headless==4.8.0.76 mediapipe==0.10.8 duckduckgo_search==3.9.6 requests==2.31.0 Pillow==10.0.0 tqdm==4.66.1
    fi
    RESULT=$?

else
    echo "Python 3.9-3.11 detected - using standard requirements"
    echo "Installing dependencies..."
    if [ -f "requirements_cloud.txt" ]; then
        pip install --user --only-binary=:all: -r requirements_cloud.txt
    else
        pip install --user --only-binary=:all: numpy==1.24.3 opencv-python-headless==4.8.0.76 mediapipe==0.10.8 duckduckgo_search==3.9.6 requests==2.31.0 Pillow==9.5.0 tqdm==4.66.1
    fi
    RESULT=$?
fi

# Check installation result
if [ $RESULT -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
    echo "Now you can run: python cloud_mediapipe_scraper.py"
else
    echo "✗ Error installing dependencies."
    echo "Try running Python directly: python cloud_mediapipe_scraper.py"
    echo "Then select option 1 for the built-in setup"
fi 