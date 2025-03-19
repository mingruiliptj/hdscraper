# Human Image Scraper: Quick Start

## Quick Installation

The easiest way to install dependencies is to use our installation script:

```bash
# Make the script executable
chmod +x install_dependencies.sh

# Run the installer
./install_dependencies.sh
```

This script will:
1. Detect your Python version
2. Install the appropriate dependencies
3. Guide you through any necessary next steps

## After Installation

Once dependencies are installed, run the scraper:

```bash
python cloud_mediapipe_scraper.py
```

Choose option 2 when prompted to run the scraper.

## Manual Installation Options

If you prefer to install manually or the script doesn't work for your environment, see the detailed instructions in `INSTALLATION.md`.

## Troubleshooting

If you encounter errors about building NumPy, particularly on Google Cloud Shell with Python 3.12+, try:

```bash
# Install NumPy with binary-only mode
pip install --user --only-binary=:all: numpy==1.26.4

# Then install the rest of the dependencies
pip install --user --only-binary=:all: opencv-python-headless mediapipe duckduckgo_search requests pillow tqdm
``` 