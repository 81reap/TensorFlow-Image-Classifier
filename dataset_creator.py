# --------------------------------------------------------------------------- #
# Author: Prayag Bhakar
# Usage :
#  $ python dataset_creator.py
# --------------------------------------------------------------------------- #

from bing_image_downloader import downloader

# --- Global Vars --- # 
output_dir = "dataset"
min_images = 1000
classes = ["dogs", "cats", "parrot"]

# --- Downloader --- #
for item in classes :
  downloader.download(item, limit=min_images, output_dir=output_dir, adult_filter_off=False, force_replace=False, timeout=60)
