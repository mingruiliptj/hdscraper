# Human Image Scraper for Google Colab

A tool to automatically download high-quality images of human subjects, process them to 1024x1024 resolution, detect faces, and save them to Google Drive.

## Features

- Downloads high-resolution images of human subjects using DuckDuckGo search
- Processes images to exactly 1024x1024 pixels while maintaining aspect ratio
- Uses MediaPipe for fast and reliable face detection
- Centers crops on detected faces whenever possible
- Saves processed images to Google Drive
- Compatible with Google Colab
- **Fixes NumPy binary incompatibility issues** commonly encountered in Colab

## How to Use in Google Colab

### Option 1: Use the Notebook

1. Upload `human_images_notebook.ipynb` to Google Colab
2. Run the notebook
3. Follow the prompts to set up the environment and run the scraper

### Option 2: Use the Python Script

1. Upload `mediapipe_scraper.py` to Google Colab
2. Run the following command in a code cell:
   ```python
   %run mediapipe_scraper.py
   ```
3. Choose option 1 to set up the environment (do this first)
4. After setup is complete, run the script again and choose option 2
5. Enter your search parameters:
   - Main keyword (e.g., "human")
   - Sub-keywords (e.g., "profile, face, portrait")
   - Project name (for the folder in Google Drive)
   - Number of images per sub-keyword

## Technical Notes

- The script uses MediaPipe instead of dlib/face_recognition for face detection
  - This avoids installation issues in Google Colab
  - Provides reliable face detection without CUDA dependencies
- Specific versions of NumPy and OpenCV are installed to avoid binary incompatibility
- Images are saved to: `/content/drive/MyDrive/Loras/[project_name]/dataset/`
- Only images with a minimum resolution of 1024x1024 pixels are processed

## Files Included

- `mediapipe_scraper.py` - Main script with NumPy binary compatibility fixes
- `colab_setup_cell.txt` - Code cell you can copy into Colab to download and run the script
- `human_images_notebook.ipynb` - Ready-to-use Colab notebook

## Requirements

The script automatically installs all required dependencies:
- requests
- Pillow
- tqdm
- duckduckgo_search
- numpy (specific version 1.24.3)
- opencv-python-headless (specific version 4.8.0.76)
- mediapipe 