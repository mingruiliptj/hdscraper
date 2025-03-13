import os
import requests
from PIL import Image
from io import BytesIO
from googleapiclient.discovery import build
from duckduckgo_search import DDGS
import time
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

class ImageScraper:
    def __init__(self, api_key=None):
        self.google_api_key = api_key
        self.target_size = (1024, 1024)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_save_directory(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

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

    def process_image(self, image_url, save_path, index):
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return False

            # Open image and convert to RGB
            image = Image.open(BytesIO(response.content)).convert('RGB')
            width, height = image.size

            # Skip if image is too small - we only want larger images
            if width < 1024 or height < 1024:
                self.logger.debug(f"Skipping small image {width}x{height}: {image_url}")
                return False

            # Skip if image isn't significantly larger than target size
            if width < 1200 or height < 1200:
                self.logger.debug(f"Skipping image not large enough {width}x{height}: {image_url}")
                return False

            # Crop center of the image to exactly 1024x1024
            cropped_image = self.crop_center(image)
            
            # Double check we have exactly 1024x1024
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

    def scrape_images(self, keyword, save_folder, num_images):
        """
        Main function to scrape and process images
        """
        save_path = self.create_save_directory(save_folder)
        self.logger.info(f"Saving images to: {save_path}")
        self.logger.info("Note: Only processing images larger than 1200x1200 pixels")
        
        # Collect image URLs from different sources
        image_urls = self.search_duckduckgo(f"buddhism {keyword}", num_images * 3)
        self.logger.info(f"Found {len(image_urls)} potential images to process")

        # Process images with progress bar
        successful_downloads = 0
        with tqdm(total=num_images) as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for i, url in enumerate(image_urls):
                    if successful_downloads >= num_images:
                        break
                        
                    if self.process_image(url, save_path, i):
                        successful_downloads += 1
                        pbar.update(1)

        self.logger.info(f"Successfully downloaded {successful_downloads} images")
        return successful_downloads

def main():
    # Example usage
    scraper = ImageScraper()
    keyword = input("Enter search keyword: ")
    save_folder = input("Enter save folder name: ")
    num_images = int(input("Enter number of images to download: "))
    
    scraper.scrape_images(keyword, save_folder, num_images)

if __name__ == "__main__":
    main() 