# --------------------------------------------------------------------------- #
# Author: Prayag Bhakar
# Usage : 
#  $ python clean_data.py [name of folder]
# --------------------------------------------------------------------------- #

# --- Imports --- #
import os
import sys
import argparse
from PIL import Image
from math import floor
from tqdm import trange
from random import shuffle
from shutil import rmtree, copyfile
# --- Imports --- #

# --- Grab Sys Args --- #
parser = argparse.ArgumentParser(description='''This script is used to clean the dataset and prepare it for image classification. This will take the image dataset and split it into 80% training images and 20% testing images. All images will be renamed, organized, and turned into 500px square images for faster training. All images with issues will be sent to a separate junk folder.''')
parser.add_argument('source', metavar='source', type=str, help='source folder')
parser.add_argument('-s', '--size', help="Max size of square images", type=int, default=500)
parser.add_argument('-t', '--train', help="Training percent", type=float, default=0.8)
parser.add_argument('-tf', '--train_folder', help="Training Folder", type=str, default='train_dataset')
parser.add_argument('-ef', '--eval_folder', help="Evaluation Folder", type=str, default='eval_dataset')
args = parser.parse_args()
# --- Grab Sys Args --- #

# --- Global Vars --- #
# get the percent cutoffs
train_percent = args.train
eval_percent = 1 - train_percent

# output square image px size
edge = args.size

# get all the different folder names
src_folder = args.source
train_folder = args.train_folder
eval_folder = args.eval_folder
# --- Global Vars --- #

# --- Update Directories --- #
# set all the path vars
path = os.getcwd()
src_path = os.path.join(path, src_folder)
train_path = os.path.join(path, train_folder)
eval_path = os.path.join(path, eval_folder)
junk_path = os.path.join(path, "junk")

# different categories
directories = [dI 
	for dI in os.listdir(src_path) 
		if os.path.isdir(os.path.join(src_path,dI))]

# replace destination folders with empty ones
if os.path.exists(train_path):
	rmtree(train_path)
os.makedirs(train_path)
if os.path.exists(eval_path):
	rmtree(eval_path)
os.makedirs(eval_path)
if os.path.exists(junk_path):
	rmtree(junk_path)
os.makedirs(junk_path)
# --- Update Directories --- #

# --- Clean Directories --- #
for sub in directories:
	# make a new sub directory in all three of the new folders
	os.makedirs(os.path.join(train_path, sub))
	os.makedirs(os.path.join(eval_path, sub))

	# get a list of all the files in the source
	files_all = os.listdir(os.path.join(src_path, sub))
	# remove hidden files
	files_all = [ file for file in files_all if not file.startswith('.')]
	# randomize the files
	shuffle(files_all)

	# calculate the number of files that go into the training folder
	num_train_images = floor(len(files_all) * train_percent)

	# count of the files transferred
	count = 0

	# copy over the training images from the source folder
	for i in trange(len(files_all), desc=sub):
		name = os.path.join(src_path, sub, files_all[i])

		img_name = sub +"_"+ str(count) +".jpg"
		end_path = train_path
		if (count > num_train_images) :
			end_path =eval_path

		new_name = os.path.join(end_path, sub, img_name)

		try:
			im = Image.open(name)
			im = im.convert("RGB")

			# get crop box dimensions and crop
			w, h = im.size
			s = min(w, h)
			w_start = floor((w-s)/2)
			h_start = floor((h-s)/2)
			im = im.crop((w_start, h_start, w_start+s, h_start+s))

			# resize the image
			im = im.resize((edge, edge))

			im.save(new_name)

			count = count + 1
		except:
			print(name)
			copyfile(os.path.join(src_path, sub, name), 
				os.path.join(junk_path, sub + "_" + files_all[i]))
			os.remove(name)
# --- Clean Directories --- #
