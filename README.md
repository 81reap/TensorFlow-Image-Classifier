# TensorFlow Image Classifier
These python scripts help set up a dataset and create a simple TensorFlow Image Classier.

## dataset_creator.py
This script is used to create a dataset based on classes of your choice. It uses the Bing image downloader package to create a custom image dataset. 
Script options can be found in the global vars section.

### Usage
``` bash
 $ python dataset_creator.py
```

## clean_data.py
This script is used to clean the dataset and prepare it for image classification. This will take the image dataset and split it into 80% training images and 20% testing images. All images will be renamed, organized, and turned into 500px square images for faster training. All images with issues will be sent to a separate junk folder.
Further script options can be found in the global vars sections

### Usage
``` bash
 $ python3 clean_data.py [name of folder]
```
