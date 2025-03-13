# Buddhist Image Scraper for LoRA Training

This script scrapes high-quality Buddhist-themed images and processes them to be suitable for LoRA training.

## Features
- Scrapes only high-resolution images (larger than 1200x1200)
- Crops images to exactly 1024x1024 from the center
- No resizing of smaller images (they are skipped)
- Maintains image quality with center cropping
- Multi-threaded downloading
- Progress bar and detailed logging

## Setup

1. Navigate to your virtual environment directory:
```bash
cd C:\ws\myloraenv\.venv
```

2. Activate the virtual environment:
```bash
# On Windows
Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated
2. Run the script:
```bash
python image_scraper.py
```

3. When prompted, enter:
   - Search keyword (will be combined with "buddhism")
   - Save folder name
   - Number of images to download

## Notes
- Only images larger than 1200x1200 pixels will be processed
- Images are center-cropped to exactly 1024x1024
- Smaller images are automatically skipped
- All images are saved in high-quality JPEG format 