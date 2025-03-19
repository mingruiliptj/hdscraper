import nbformat as nbf
import re

def create_notebook_from_image_scraper():
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Title and description markdown cell
    nb.cells.append(nbf.v4.new_markdown_cell("""
# Human Image Scraper for Machine Learning Datasets

This notebook allows you to scrape images of humans with various attributes to create training datasets.
It has built-in face detection to center crops on faces, and produces 1024x1024 pixel images suitable for machine learning.
All images are saved directly to your Google Drive in the structure: `/content/drive/MyDrive/Loras/[project_name]/dataset/`
    """))
    
    # Installation cell
    nb.cells.append(nbf.v4.new_code_cell("""
# Check if running in Colab
import sys
IN_COLAB = 'google.colab' in sys.modules
print(f"Running in Google Colab: {IN_COLAB}")

# Mount Google Drive
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted at /content/drive")

# Install required packages
!pip install -q requests
!pip install -q Pillow
!pip install -q google-api-python-client
!pip install -q duckduckgo-search
!pip install -q tqdm

# Install face_recognition (more complex in Colab)
if IN_COLAB:
    !apt-get -qq install -y libsm6 libxext6 libxrender-dev libglib2.0-0
    !pip install -q dlib
    !pip install -q face_recognition
else:
    !pip install -q face_recognition

!pip install -q numpy

# Display versions for debugging
!pip list | grep -E "requests|Pillow|google|duckduckgo|tqdm|face|numpy|dlib"
    """))
    
    # Import cell
    nb.cells.append(nbf.v4.new_code_cell("""
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
import face_recognition
import numpy as np
from google.colab import files
    """))
    
    # Modify the class code to update the create_save_directory method
    with open('image_scraper.py', 'r') as f:
        content = f.read()
    
    # Replace the create_save_directory method in the content
    modified_content = re.sub(
        r'def create_save_directory\(self, save_folder\):.*?return save_folder',
        """def create_save_directory(self, project_name):
        # Create directory structure: Loras/project_name/dataset
        save_path = os.path.join('/content/drive/MyDrive/Loras', project_name, 'dataset')
        os.makedirs(save_path, exist_ok=True)
        self.logger.info(f"Saving images to Google Drive path: {save_path}")
        return save_path""",
        content,
        flags=re.DOTALL
    )
    
    # Extract the modified ImageScraper class
    class_match = re.search(r'class ImageScraper.*?(?=def main\(\)|$)', modified_content, re.DOTALL)
    if class_match:
        class_code = class_match.group(0)
        # Fix indentation for notebook
        class_code = "\n".join([line for line in class_code.split('\n')])
        nb.cells.append(nbf.v4.new_code_cell(class_code))
    
    # Create an instance and run cell
    nb.cells.append(nbf.v4.new_code_cell("""
# Create an instance of the scraper
scraper = ImageScraper()
    """))
    
    # Input parameters cell - change save_folder to project_name
    nb.cells.append(nbf.v4.new_code_cell("""
# Set parameters for image scraping
main_keyword = input("Enter main keyword (e.g., 'human'): ")
sub_keywords = input("Enter sub-keywords separated by commas (e.g., 'profile, face, portrait'): ")
project_name = input("Enter project name (folder will be created in Google Drive): ")
num_images_per_keyword = int(input("Enter number of images to download per sub-keyword: "))
    """))
    
    # Run scraper cell - update to use project_name
    nb.cells.append(nbf.v4.new_code_cell("""
# Run the scraper
total_images = scraper.scrape_images(main_keyword, sub_keywords, project_name, num_images_per_keyword)
print(f"Total images downloaded: {total_images}")
print(f"Images saved to: /content/drive/MyDrive/Loras/{project_name}/dataset/")
    """))
    
    # Write the notebook to a file
    with open('Human_Image_Scraper_Colab.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook created successfully: Human_Image_Scraper_Colab.ipynb")

if __name__ == "__main__":
    create_notebook_from_image_scraper() 