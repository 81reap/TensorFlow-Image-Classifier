# TensorFlow Image Classifier
These python scripts help set up a dataset and create a simple TensorFlow Image Classier. These scripts work best with an Nvida GPU.

## dataset_creator.py
This script is used to create a dataset based on class of your choice. It uses the Bing image downloader package to search up the term and add relevant images to the class.

``` bash
Usage
 $ python dataset_creator.py [class term]
Example
 $ python dataset_creator.py apple
 $ python dataset_creator.py orange
 $ python dataset_creator.py cherry
```

## clean_data.py
This script is used to clean the dataset and prepare it for image classification. This will take the image dataset and split it into 80% training images and 20% testing images. All images will be renamed, organized, and turned into 500px square images for faster training. All images with issues will be sent to a separate junk folder.

``` bash
Usage
 $ python clean_data.py [name of folder]
Example
 $ python clean_data.py dataset
```

## create_CNN_model.py
This script will create and test an image classfication model based on a custom Convolutional Neural Network (CNN) model. It will output the modle in multiple useable forms as well as graphs of the traning the the confusiton matrix. The text file contains the class name of each output tensor in order of the tensor.

``` bash
Usage
 $ python create_CNN_model.py [version]
Example
 $ python create_CNN_model.py 1.0
```

# Licence

BSD 3-Clause License

Copyright (c) 2019, Prayag Bhakar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

