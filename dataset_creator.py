# --------------------------------------------------------------------------- #
# Author: Prayag Bhakar
# --------------------------------------------------------------------------- #

# --- Imports --- #
import sys
import argparse
from bing_image_downloader import downloader
# --- Imports --- #

# --- Grab Sys Args --- #
parser = argparse.ArgumentParser(description='''This script is used to create a dataset based on classes of your choice. It uses the Bing image downloader package to create a custom image dataset.''')
parser.add_argument('term', metavar='term', type=str, help='the class name to search up and download images for')
parser.add_argument('-o', '--output', help='Output file directory', type=str, default='dataset')
parser.add_argument('-m', '--min', help='Minimum number of images', type=int, default=1000)
args = parser.parse_args()
# --- Grab Sys Args --- #

# --- Global Vars --- # 
term = args.term
output_dir = args.output
min_images = args.min
# --- Global Vars --- #

# --- Downloader --- #
downloader.download(term, limit=min_images, output_dir=output_dir, adult_filter_off=False, force_replace=False, timeout=60)
# --- Downloader --- # 