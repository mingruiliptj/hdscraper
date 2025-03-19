"""
# Human Image Scraper for LoRA Training - CUDA ERROR FIXED VERSION

This script scrapes high-quality human images with face detection using CPU-only mode.
It will download images and create 1024x1024 crops centered on faces.
All images are saved to Google Drive.

## IMPORTANT: Run the setup cell FIRST to avoid CUDA errors
"""

# -------------------------------------------------
# SETUP CELL - RUN THIS FIRST IN COLAB
# -------------------------------------------------
"""
# Set environment variables before imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DLIB_USE_CUDA"] = "0"

# Install basic dependencies first
!pip install -q requests Pillow duckduckgo_search tqdm numpy

# Install system dependencies 
!apt-get -qq update
!apt-get -qq install -y libsm6 libxext6 libxrender-dev libglib2.0-0 cmake

# Remove any existing installations
!pip uninstall -y dlib face_recognition

# IMPORTANT: Install pre-built wheels to avoid compilation freezes
print("Installing pre-built dlib wheel...")
!pip install -q https://github.com/z-mahmud22/Dlib_Windows_Python/raw/main/dlib-19.22.99-cp310-cp310-win_amd64.whl || pip install -q dlib==19.22.1

print("Installing face_recognition...")
!pip install -q face_recognition

# Verify installation
!pip list | grep -E "dlib|face"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify dlib can be imported without CUDA issues
try:
    import dlib
    dlib.DLIB_USE_CUDA = False
    import face_recognition
    print("✅ Face recognition loaded successfully in CPU-only mode")
except Exception as e:
    print(f"❌ Error: {e}")
    print("If the pre-built wheel didn't work, try running this alternative method:")
    print("!pip install -q dlib==19.22.1")
    print("!pip install -q face_recognition")

print("✅ Setup complete - Now run the main script cell")
"""

# -------------------------------------------------
# MAIN SCRIPT - RUN THIS AFTER THE SETUP CELL
# -------------------------------------------------

# Critical: Set environment variables before ANY imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DLIB_USE_CUDA"] = "0"

import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS
import time
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

# Import dlib first and configure it to not use CUDA
try:
    import dlib
    # Override dlib's CUDA detection
    dlib.DLIB_USE_CUDA = False
    import face_recognition
    print("✅ Face recognition loaded successfully in CPU-only mode")
except Exception as e:
    print(f"❌ Error loading face_recognition: {e}")
    print("Please run the setup cell first and ensure CUDA environment variables are set correctly")
    raise

class ImageScraper:
    def __init__(self):
        self.target_size = (1024, 1024)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_save_directory(self, project_name):
        # Create directory structure: Loras/project_name/dataset
        save_path = os.path.join('/content/drive/MyDrive/Loras', project_name, 'dataset')
        os.makedirs(save_path, exist_ok=True)
        self.logger.info(f"Saving images to Google Drive path: {save_path}")
        return save_path

    def crop_center(self, image):
        width, height = image.size
        
        # Calculate dimensions for center crop
        if width > height:
            left = (width - height) // 2
            top = 0
            right = left + height
            bottom = height
        else:
            top = (height - width) // 2
            left = 0
            bottom = top + width
            right = width
            
        # Get the center crop
        cropped = image.crop((left, top, right, bottom))
        
        # If the cropped image is still larger than 1024x1024, take the center 1024x1024
        if cropped.size[0] > 1024:
            size = cropped.size[0]
            margin = (size - 1024) // 2
            cropped = cropped.crop((margin, margin, margin + 1024, margin + 1024))
            
        return cropped

    def resize_keeping_aspect_ratio(self, image, target_size):
        """Resize image to target size while maintaining aspect ratio"""
        width, height = image.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1:  # width > height
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:  # height > width
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def crop_around_face(self, image):
        """Crop image to 1024x1024 keeping faces centered"""
        # Convert PIL Image to numpy array for face_recognition
        img_array = np.array(image)
        
        try:
            # Explicitly use the HOG-based model which is CPU-only
            # Use lower upsample value for better performance
            face_locations = face_recognition.face_locations(
                img_array, 
                model='hog',  # Always use HOG (CPU) model, never CNN
                number_of_times_to_upsample=1  # Lower value = faster but may miss small faces
            )
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return self.crop_center(image)
        
        if not face_locations:
            # If no faces detected, fall back to center crop
            self.logger.debug("No faces detected, using center crop")
            return self.crop_center(image)
            
        # Calculate the center point of all faces
        centers = []
        for top, right, bottom, left in face_locations:
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            centers.append((center_x, center_y))
            
        # Use the average center point of all faces
        center_x = int(sum(x for x, _ in centers) / len(centers))
        center_y = int(sum(y for _, y in centers) / len(centers))
        
        # Calculate crop box
        width, height = image.size
        crop_size = 1024
        
        # Ensure the crop box stays within image boundaries
        left = max(0, min(center_x - crop_size // 2, width - crop_size))
        top = max(0, min(center_y - crop_size // 2, height - crop_size))
        right = left + crop_size
        bottom = top + crop_size
        
        return image.crop((left, top, right, bottom))

    def process_image(self, image_url, save_path, index):
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return False

            # Open image and convert to RGB
            image = Image.open(BytesIO(response.content)).convert('RGB')
            width, height = image.size

            # Skip if image is too small
            if width < 1024 or height < 1024:
                self.logger.debug(f"Skipping small image {width}x{height}: {image_url}")
                return False

            # Resize image while maintaining aspect ratio
            # Make sure the smaller dimension is at least 1024px
            if width < height:
                new_width = 1024
                new_height = int(height * (new_width / width))
            else:
                new_height = 1024
                new_width = int(width * (new_height / height))
                
            image = self.resize_keeping_aspect_ratio(image, (new_width, new_height))

            # Crop around face to exactly 1024x1024
            cropped_image = self.crop_around_face(image)
            
            # Verify final size
            if cropped_image.size != (1024, 1024):
                self.logger.error(f"Unexpected crop size {cropped_image.size}")
                return False

            # Generate unique filename based on image content
            image_hash = hashlib.md5(response.content).hexdigest()[:10]
            filename = f"image_{index}_{image_hash}.jpg"
            save_path = os.path.join(save_path, filename)
            
            # Save the image with high quality
            cropped_image.save(save_path, "JPEG", quality=95)
            self.logger.info(f"Saved image {filename} (original size: {width}x{height})")
            return True

        except Exception as e:
            self.logger.error(f"Error processing image {image_url}: {str(e)}")
            return False

    def search_duckduckgo(self, keyword, max_results):
        image_urls = []
        try:
            with DDGS() as ddgs:
                results = ddgs.images(
                    keyword,
                    max_results=max_results * 3  # Get more results since we're being more selective
                )
                for r in results:
                    if r['image']:
                        image_urls.append(r['image'])
        except Exception as e:
            self.logger.error(f"Error searching DuckDuckGo: {str(e)}")
        return image_urls

    def scrape_images(self, main_keyword, sub_keywords, project_name, num_images_per_keyword):
        """
        Main function to scrape and process images
        """
        save_path = self.create_save_directory(project_name)
        self.logger.info(f"Saving images to: {save_path}")
        self.logger.info("Note: Only processing images with minimum 1024x1024 pixels")
        
        total_successful_downloads = 0
        # Split sub-keywords and process each one
        sub_keyword_list = [sk.strip() for sk in sub_keywords.split(',')]
        self.logger.info(f"Processing {len(sub_keyword_list)} sub-keywords: {sub_keyword_list}")
        
        for sub_keyword in sub_keyword_list:
            if not sub_keyword:
                continue
                
            search_query = f"{main_keyword} {sub_keyword}"
            self.logger.info(f"Searching for: {search_query}")
            
            # Collect image URLs from different sources
            image_urls = self.search_duckduckgo(search_query, num_images_per_keyword * 3)
            self.logger.info(f"Found {len(image_urls)} potential images for '{search_query}'")

            # Process images with progress bar
            successful_downloads = 0
            with tqdm(total=num_images_per_keyword, desc=f"Processing '{sub_keyword}'") as pbar:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for i, url in enumerate(image_urls):
                        if successful_downloads >= num_images_per_keyword:
                            break
                            
                        if self.process_image(url, save_path, f"{sub_keyword}_{i}"):
                            successful_downloads += 1
                            total_successful_downloads += 1
                            pbar.update(1)

            self.logger.info(f"Downloaded {successful_downloads} images for '{search_query}'")
        
        self.logger.info(f"Total successfully downloaded images: {total_successful_downloads}")
        return total_successful_downloads

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Human Image Scraper for LoRA Training (CPU-only version)")
    print("This script will download high-quality human images with face detection.")
    print("Images will be saved to your Google Drive in Loras/[project_name]/dataset/")
    print("=" * 80)
    
    # Initialize the scraper
    scraper = ImageScraper()
    
    # Set your parameters
    main_keyword = input("Enter main keyword (e.g., 'human'): ")
    sub_keywords = input("Enter sub-keywords separated by commas (e.g., 'profile, face, portrait'): ")
    project_name = input("Enter project name (folder will be created in Google Drive): ")
    num_images_per_keyword = int(input("Enter number of images to download per sub-keyword: "))
    
    # Start scraping
    scraper.scrape_images(main_keyword, sub_keywords, project_name, num_images_per_keyword)
    
    print(f"\nImages saved to: /content/drive/MyDrive/Loras/{project_name}/dataset/") 