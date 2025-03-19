#!/usr/bin/env python3
"""
# Human Image Scraper with MediaPipe Face Detection for Google Cloud
# This version fixes the NumPy binary incompatibility error

To use this script in Google Cloud:
1. Run: python cloud_mediapipe_scraper.py
2. Choose option 1 to set up the environment first
3. Once setup is complete, run the script again and choose option 2

Images will be saved to the mapped Google Drive at: ~/google-drive/Loras/[project_name]/dataset/
"""

def setup_environment():
    """Set up the environment with proper dependency installation order"""
    import os
    import sys
    import subprocess
    import platform
    
    print("\n========== Setting Up Environment ==========")
    print("This may take a few minutes...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 13:
        print("\nWARNING: Python 3.13+ detected. Some packages like mediapipe may not be compatible.")
        print("Consider using Python 3.10-3.12 for full compatibility.")
        
        user_choice = input("Continue with limited functionality? (y/n): ")
        if user_choice.lower() != 'y':
            print("Setup aborted. Please use a compatible Python version (3.10-3.12 recommended).")
            return False
            
        # For newer Python versions, try to install compatible packages
        print("Installing compatible packages for Python 3.13+...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                                  "numpy>=2.1.0", "opencv-python-headless>=4.8.0", 
                                  "requests", "Pillow", "tqdm", "duckduckgo-search>=3.9.6"])
            print("Note: MediaPipe is not available for Python 3.13+. Face detection will be limited.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False
    elif python_version.major == 3 and python_version.minor >= 12:
        # Python 3.12 compatibility fix
        print("\nPython 3.12 detected. Using compatibility fixes...")
        
        # Clean up any potentially conflicting packages
        print("Removing potentially conflicting packages...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                       "numpy", "opencv-python", "opencv-python-headless", 
                       "mediapipe", "duckduckgo_search", "setuptools"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # First install a compatible setuptools version to fix the pkgutil.ImpImporter error
        print("Installing compatible setuptools...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "setuptools==68.2.2"])
        
        # Install NumPy
        print("Installing NumPy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.0"])
        
        # Install basic dependencies
        print("Installing basic dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                              "requests", "Pillow", "tqdm"])
        
        # Install OpenCV first, then mediapipe
        print("Installing OpenCV...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless==4.8.0.76"])
        
        print("Installing MediaPipe...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.13"])
        
        # Install DuckDuckGo search
        print("Installing DuckDuckGo search library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckduckgo_search==3.9.6"])
    else:
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.13"])
        
        # Install DuckDuckGo search
        print("Installing DuckDuckGo search library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckduckgo_search==3.9.6"])
    
    # Verify installations
    try:
        import numpy
        import cv2
        print(f"NumPy version: {numpy.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
        
        try:
            import mediapipe
            print(f"MediaPipe version: {mediapipe.__version__}")
            has_mediapipe = True
        except ImportError:
            print("MediaPipe not available on this Python version.")
            has_mediapipe = False
            
        try:
            from duckduckgo_search import DDGS
            print("DuckDuckGo search library installed.")
        except ImportError as e:
            print(f"Error importing duckduckgo_search: {e}")
            print("Try installing with: pip install duckduckgo-search")
            return False
            
        print("\n✓ Dependencies installed successfully!")
    except ImportError as e:
        print(f"Error importing a dependency: {e}")
        return False
    
    # Verify Google Drive path
    gdrive_path = os.path.expanduser("~/google-drive/Loras")
    if os.path.exists(gdrive_path):
        print(f"\nVerified Google Drive path: {gdrive_path}")
    else:
        print(f"\nWarning: Google Drive path not found at {gdrive_path}")
        print("Please ensure the Google Drive is properly mounted")
    
    print("\nSetup complete!")
    if not has_mediapipe:
        print("NOTE: MediaPipe is not available. Face detection will be limited.")
    print("Please run the script again and select option 2 to scrape images.")
    return True

def run_scraper():
    """Run the image scraper with MediaPipe face detection"""
    try:
        import os
        import requests
        from PIL import Image
        from io import BytesIO
        import time
        from tqdm import tqdm 
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        import logging
        import numpy as np
        import cv2
        
        # Check if mediapipe is available
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short range, 1 for long range
                min_detection_confidence=0.5
            )
            has_mediapipe = True
        except ImportError:
            print("MediaPipe not available. Using fallback face detection.")
            has_mediapipe = False
            
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            try:
                # Try alternative import style
                from duckduckgo_search.duckduckgo_search import DDGS
            except ImportError:
                print("Error: DuckDuckGo search library not found.")
                print("Please run setup option 1 first.")
                return False
        
        # If we made it here, imports are successful
        print("\nAll dependencies loaded successfully!")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please run option 1 first to set up the environment.")
        return False
    
    class ImageScraper:
        def __init__(self):
            self.target_size = (1024, 1024)
            self.setup_logging()
            self.has_mediapipe = has_mediapipe

        def setup_logging(self):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

        def create_save_directory(self, project_name):
            # Create directory structure in mounted Google Drive: ~/google-drive/Loras/project_name/dataset
            save_path = os.path.expanduser(f"~/google-drive/Loras/{project_name}/dataset")
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
            
            try:
                # Use LANCZOS resampling if available
                return image.resize((new_width, new_height), Image.LANCZOS)
            except AttributeError:
                # Fall back to BICUBIC for newer Pillow versions
                return image.resize((new_width, new_height), Image.BICUBIC)

        def crop_around_face(self, image):
            """Crop image to 1024x1024 keeping faces centered using MediaPipe if available"""
            try:
                # If mediapipe is not available, fall back to center crop
                if not self.has_mediapipe:
                    self.logger.debug("MediaPipe not available, using center crop")
                    return self.crop_center(image)
                
                # Convert PIL Image to CV2 format
                img_array = np.array(image)
                
                # Handle grayscale images by converting to RGB
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
                # Convert to RGB for MediaPipe (it expects RGB)
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
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
                full_save_path = os.path.join(save_path, filename)
                
                # Save the image with high quality
                cropped_image.save(full_save_path, "JPEG", quality=95)
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
                    # Process sequentially to avoid overloading the server with requests
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
    print("Using MediaPipe for reliable face detection in Google Cloud")
    print("Images will be saved to your mapped Google Drive at: ~/google-drive/Loras/[project_name]/dataset/")
    print("=" * 80)
    
    # Check if Google Drive is mounted
    gdrive_path = os.path.expanduser("~/google-drive/Loras")
    if not os.path.exists(gdrive_path):
        print(f"\nWARNING: Google Drive path not found at {gdrive_path}")
        print("Please ensure Google Drive is properly mounted.")
        create_dir = input("Do you want to create this directory? (y/n): ")
        if create_dir.lower() == 'y':
            os.makedirs(gdrive_path, exist_ok=True)
            print(f"Created directory: {gdrive_path}")
        else:
            print("Please mount Google Drive correctly and try again.")
            return False
    
    # Initialize the scraper
    scraper = ImageScraper()
    
    # Set your parameters
    main_keyword = input("Enter main keyword (e.g., 'human'): ")
    sub_keywords = input("Enter sub-keywords separated by commas (e.g., 'profile, face, portrait'): ")
    project_name = input("Enter project name (folder will be created in Google Drive): ")
    
    try:
        num_images_per_keyword = int(input("Enter number of images to download per sub-keyword: "))
    except ValueError:
        print("Invalid number. Using default of 10 images per keyword.")
        num_images_per_keyword = 10
    
    # Start scraping
    scraper.scrape_images(main_keyword, sub_keywords, project_name, num_images_per_keyword)
    
    print(f"\nImages saved to: ~/google-drive/Loras/{project_name}/dataset/")
    return True

if __name__ == "__main__":
    print("\n========== Human Image Scraper for Google Cloud ==========")
    print("1. Set up the environment (do this first)")
    print("2. Run the image scraper (do this after setup)")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == "1":
        setup_environment()
    elif choice == "2":
        run_scraper()
    else:
        print("Invalid choice. Please enter 1 or 2.") 