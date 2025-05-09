"""
# Human Image Scraper with MediaPipe Face Detection for Google Colab
# This version fixes the NumPy binary incompatibility error

To use this script in Google Colab:
1. Upload this file to Google Colab
2. Run in a code cell: %run mediapipe_scraper.py
3. Choose option 1 to set up the environment
4. Once setup is complete, run the script again and choose option 2

This script handles:
- NumPy version incompatibility issues
- Binary dependency conflicts in Colab
- Proper package installation order
"""

def setup_environment():
    """Set up the environment with proper dependency installation order"""
    import os
    import sys
    import subprocess
    
    # Check if in Colab
    IN_COLAB = 'google.colab' in sys.modules
    if not IN_COLAB:
        print("Warning: This script is designed for Google Colab")
        return False
        
    print("\n========== Setting Up Environment ==========")
    print("This may take a few minutes...")
    
    # Clean up any potentially conflicting packages
    print("Removing potentially conflicting packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                   "numpy", "opencv-python", "opencv-python-headless", 
                   "mediapipe", "duckduckgo_search"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # First install NumPy (critical for compatibility)
    print("Installing NumPy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.24.3"])
    
    # Install basic dependencies
    print("Installing basic dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "requests", "Pillow", "tqdm"])
    
    # Install OpenCV first, then mediapipe
    print("Installing OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless==4.8.0.76"])
    
    print("Installing MediaPipe...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.8"])
    
    # Install DuckDuckGo search
    print("Installing DuckDuckGo search library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckduckgo_search==3.9.6"])
    
    # Verify installations
    try:
        import numpy
        import cv2
        import mediapipe
        from duckduckgo_search import DDGS
        print("\n✓ All dependencies installed successfully!")
        print(f"NumPy version: {numpy.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"MediaPipe version: {mediapipe.__version__}")
    except ImportError as e:
        print(f"Error importing a dependency: {e}")
        return False
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("Please ensure you're running this in Google Colab")
    
    print("\nSetup complete - MediaPipe face detection is ready!")
    print("Please run the script again and select option 2 to scrape images.")
    return True

def run_scraper():
    """Run the image scraper with MediaPipe face detection"""
    try:
        import os
        import requests
        from PIL import Image
        from io import BytesIO
        from duckduckgo_search import DDGS
        import time
        from tqdm.notebook import tqdm
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        import logging
        import numpy as np
        import cv2
        import mediapipe as mp
        
        # If we made it here, imports are successful
        print("\nAll dependencies loaded successfully!")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please run option 1 first to set up the environment.")
        return False
    
    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short range, 1 for long range
        min_detection_confidence=0.5
    )
    
    class ImageScraper:
        def __init__(self, enable_img_resize=True):
            self.target_size = (1024, 1024)
            self.setup_logging()
            self.enable_img_resize = enable_img_resize

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
                
            return image.resize((new_width, new_height), Image.LANCZOS)

        def crop_around_face(self, image):
            """Crop image to 1024x1024 keeping faces centered using MediaPipe"""
            try:
                # Convert PIL Image to CV2 format
                img_array = np.array(image)
                
                # Handle grayscale images by converting to RGB
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
                # Convert to RGB for MediaPipe (it expects RGB)
                if img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Drop alpha channel
                
                # Process with MediaPipe
                results = face_detection.process(img_array)
                
                if not results.detections:
                    # If no faces detected, fall back to center crop
                    return self.crop_center(image)
                    
                # Calculate the center point of all faces
                height, width = img_array.shape[:2]
                centers = []
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Calculate center of face
                    center_x = x + w // 2
                    center_y = y + h // 2
                    centers.append((center_x, center_y))
                    
                # Use the average center point of all faces
                center_x = int(sum(x for x, _ in centers) / len(centers))
                center_y = int(sum(y for _, y in centers) / len(centers))
                
                # Calculate crop box
                img_width, img_height = image.size
                crop_size = 1024
                
                # Ensure the crop box stays within image boundaries
                left = max(0, min(center_x - crop_size // 2, img_width - crop_size))
                top = max(0, min(center_y - crop_size // 2, img_height - crop_size))
                right = left + crop_size
                bottom = top + crop_size
                
                return image.crop((left, top, right, bottom))
            except Exception as e:
                self.logger.error(f"Error in face detection: {e}")
                return self.crop_center(image)

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

            
                # Crop around face to exactly 1024x1024 if enabled
                if self.enable_img_resize:

                    # Resize image while maintaining aspect ratio
                    # Make sure the smaller dimension is at least 1024px
                    if width < height:
                        new_width = 1024
                        new_height = int(height * (new_width / width))
                    else:
                        new_height = 1024
                        new_width = int(width * (new_height / height))
                    image = self.resize_keeping_aspect_ratio(image, (new_width, new_height))

                    cropped_image = self.crop_around_face(image)
                    
                    # Verify final size
                    if cropped_image.size != (1024, 1024):
                        self.logger.error(f"Unexpected crop size {cropped_image.size}")
                else:
                    cropped_image = image

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

    print("=" * 80)
    print("Human Image Scraper for LoRA Training")
    print("Using MediaPipe for reliable face detection in Google Colab")
    print("Images will be saved to your Google Drive in Loras/[project_name]/dataset/")
    print("=" * 80)
    
    # Set your parameters
    main_keyword = input("Enter main keyword (e.g., 'human'): ")
    sub_keywords = input("Enter sub-keywords separated by commas (e.g., 'profile, face, portrait'): ")
    project_name = input("Enter project name (folder will be created in Google Drive): ")
    num_images_per_keyword = int(input("Enter number of images to download per sub-keyword: "))

    # Ask user whether to enable cropping
    crop_choice = input("Enable image resizing? ('yes' or 'no'): ")
    enable_img_resize = crop_choice.lower() == 'yes'

    # Initialize the scraper
    scraper = ImageScraper(enable_img_resize=enable_img_resize)

    # Start scraping
    scraper.scrape_images(main_keyword, sub_keywords, project_name, num_images_per_keyword)

    print(f"\nImages saved to: /content/drive/MyDrive/Loras/{project_name}/dataset/")
    return True

if __name__ == "__main__":
    print("\n========== Human Image Scraper for Google Colab ==========")
    print("1. Set up the environment (do this first)")
    print("2. Run the image scraper (do this after setup)")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == "1":
        setup_environment()
    elif choice == "2":
        run_scraper()
    else:
        print("Invalid choice. Please enter 1 or 2.")
