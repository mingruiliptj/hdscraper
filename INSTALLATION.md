# Installation Guide

## Python Version Compatibility

This project works best with Python 3.10-3.12. There are known compatibility issues with Python 3.13+ for some dependencies, especially MediaPipe.

## Installation Options

### Option 1: For Google Cloud Shell with Python 3.9-3.11

```bash
# Install directly using the cloud-optimized requirements
pip install --user -r requirements_cloud.txt
```

### Option 2: For Google Cloud Shell with Python 3.12

If you're running Python 3.12 in Google Cloud Shell, use this specialized requirements file:

```bash
# For Python 3.12 environments in Google Cloud
pip install --user -r requirements_cloud_py312.txt
```

### Option 3: For Python 3.13+

Some features (particularly face detection with MediaPipe) may be limited:

```bash
# Install compatible packages for newer Python versions
pip install numpy>=2.1.0 opencv-python-headless>=4.8.0 requests pillow tqdm duckduckgo-search>=3.9.6
```

### Option 4: Using the Script's Built-in Setup

Run the script and choose option 1 to set up the environment:

```bash
python cloud_mediapipe_scraper.py
# Then select option 1
```

## Troubleshooting Common Issues

### "No matching distribution found for mediapipe"
- This error occurs on Python 3.13+ as MediaPipe doesn't support these versions yet.
- Solution: Use Python 3.10-3.12 or continue without MediaPipe (using center cropping instead of face detection).

### NumPy Build Errors in Google Cloud Shell
If you encounter build errors with NumPy in Google Cloud Shell like:
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

Try these solutions:

1. Use the Python 3.12-specific requirements file:
```bash
pip install --user -r requirements_cloud_py312.txt
```

2. Install NumPy separately first:
```bash
pip install --user --only-binary=:all: numpy==1.26.4
```

3. Create a virtual environment with a compatible Python version:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements_cloud.txt
```

### Permissions Issues
- If you encounter permissions errors:
  ```bash
  pip install --user -r requirements_cloud.txt
  ```

## Running the Script

After successful installation:

```bash
python cloud_mediapipe_scraper.py
# Then select option 2 to run the scraper
```

Images will be saved to: `~/google-drive/Loras/[project_name]/dataset/` 