# --------------------------------------------------------------------------- #
# Author: Prayag Bhakar
# Usage :
#  $ python create_CNN_model.py [class term]
# --------------------------------------------------------------------------- #

# --- Imports --- #
import os
import argparse
import numpy as np
from tqdm import trange
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import lite
from pandas import DataFrame
import matplotlib.pyplot as plt
from os import listdir as ListDir
from os.path import join as JoinPath
# --- Imports --- #

# --- Grab Sys Args --- #
parser = argparse.ArgumentParser(description='''This script will create and test an image classfication model based on a custom Convolutional Neural Network (CNN) model. It will output the modle in multiple useable forms as well as graphs of the traning the the confusiton matrix. The text file contains the class name of each output tensor in order of the tensor.''')
parser.add_argument('version', metavar='version', type=str, help='Name of the model')
parser.add_argument('-tf', '--train_folder', help='Training folder name', type=str, default='train_dataset')
parser.add_argument('-ef', '--eval_folder', help='Testing folder name', type=str, default='eval_dataset')
args = parser.parse_args()
# --- Grab Sys Args --- #


# --- Global Vars --- #
IMAGE_SIZE = 175
MODEL_NAME = "TensorFlow_CNN_[" + args.version + "]"

TEST_FOLDER = args.eval_folder
TRAIN_FOLDER = args.train_folder
TMP_MODEL = "tmp_max.h5"
ROOT_DIR = os.getcwd()
IMG_NAME = MODEL_NAME+"_train.png"
KERAS_MODEL = MODEL_NAME+".h5"
TFLITE_MODEL = MODEL_NAME+".tflite"
TFJS_MODEL = ROOT_DIR + MODEL_NAME + ".json"
LABEL_MODEL = MODEL_NAME+".txt"
CONFUSION_NAME = MODEL_NAME+"_confusion.png"
# --- Global Vars --- #

# --- Setup the Dataset Generators --- #
## 1. create dataset for the training and testing images
## 2. Use the datasets to create generators to pipeline to TensorFlow

classes = ListDir(TEST_FOLDER)
classes.sort()

def getDF(folder):
  # get all the classes in the images folder
  classes = ListDir(folder)
  df_data = DataFrame()

  for c in classes:
    # Get a list of all images in the class folder.
    imgs = ListDir(JoinPath(folder, c))

    # Create the dataframe.
    df2 = DataFrame(imgs, columns=['image_id'])
    df2['class'] = c 

    df_data = df_data.append(df2)
  
  return df_data

df_train = getDF(TRAIN_FOLDER)
df_train['class'].value_counts()

df_test = getDF(TEST_FOLDER)
df_test['class'].value_counts()

num_train_samples = len(df_train)
num_val_samples = len(df_test)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.0,0.4),
    horizontal_flip=True,
    vertical_flip=True
)

train_gen = datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=train_batch_size,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    TEST_FOLDER,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=val_batch_size,
    class_mode='categorical'
)

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(
    TEST_FOLDER,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
# --- Setup the Dataset Generators --- #

# --- Create the Model --- #
# NOTE: the input shape must match the data being sent through. In this case,
#  we are sending a 3D array with the first two lengths being of size s and the
#  last length being 1 to represent the 1 byte colour of greyscale
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))

# Each convolution and pooling layer helps specify how the model looks at the
#  picture and how it filters information to better understand what is going on
#
# 2D Convolutions - these look at the data that has been passed through and
#  average the values found in a 3x3 square portion of the 2D matrix. These
#  averages cross over each other and highlight certain details while filtering
#  out others
# Learn more: https://bit.ly/2JjPPps
model.add(tf.keras.layers.Conv2D(
    filters = 32, 
    kernel_size = (3,3), 
    activation = 'relu' ))
model.add(tf.keras.layers.Conv2D(
    filters = 32, 
    kernel_size = (3,3), 
    activation = 'relu' ))
# 2D Max Pooling - these look at the data that has been passed through and take
#  the max value found in a 2x2 square portion of the 2D matrix. These 2x2
#  square portions do not cross over each other and cause squares with
#  highlighted details to be passed through this filtration step and mutes non
#  important details 
# Learn more: https://bit.ly/2zFokBc
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2))) 
# Dropout - these drop certian nodes in the model during training. This helps
#  limit the amount of overfitting of the testing data
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(
    filters = 64, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.Conv2D(
    filters = 64, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(
    filters = 128, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.Conv2D(
    filters = 128, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(
    filters = 256, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.Conv2D(
    filters = 256, 
    kernel_size = (3,3), 
    activation ='relu' ))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(len(classes), activation = "softmax"))

model.summary()
# --- Create the Model --- #

# --- Train the Model --- #
# compiles the model to train with with Adam optimizer and measure loss with 
#  Categorical Crossentropy
model.compile(
    tf.keras.optimizers.Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# This callback saves the model after each epoch if the model accurcy has
#  increased. If the accurcy isn't higher then the model isn't checkpoint saved.
#  This is usefull because the more you train your model, the more likly it is
#  to overfit the traing data, and in turn decreasing the vaidation accuracy.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    TMP_MODEL, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)
# This callback function reduces the learning rate of the model as it comes
#  close to its peak. This is usefull so that the model doesn't overshoot it's 
#  peak which will cause the validition accuracy to decrease
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.5, 
    patience=3, 
    verbose=1, 
    mode='max', 
    min_lr=0.00001
)
                      
callbacks_list = [checkpoint, reduce_lr]

history = model.fit(
    train_gen, 
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=40, 
    verbose=1,
    callbacks=callbacks_list
)
# --- Train the Model --- #

# --- Create the model files --- #
# --- Keras Model ---
model.load_weights(TMP_MODEL)
model.save(KERAS_MODEL)

# --- TFLite Model ---
converter = lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
tfmodel = converter.convert()
open (TFLITE_MODEL , "wb").write(tfmodel)

# --- TF.js Model ---
tfjs.converters.save_keras_model(model, ".")

# --- Labels Model ---
with open(LABEL_MODEL, 'w') as f:
    for item in classes:
        f.write("%s\n" % item)
# --- Create the model files --- #

# --- TensorFlow Lite Model Meta data --- #
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = MODEL_NAME
model_meta.description = ("Identify different types of plants")
model_meta.version = MODEL_VERSION
model_meta.author = "Prayag Bhakar [prayag.bhakar@gmail.com]"
model_meta.license = ("Revised BSD License"
                      "https://github.com/PrayagBhakar/TensorFlow-Image-Classifier/blob/master/LICENSE")

# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(IMAGE_SIZE, IMAGE_SIZE))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (_metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5] # idk what this is
input_normalization.options.std = [127.5] # idk what this is
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of teh respective plants."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename(LABEL_MODEL)
label_file.description = "Plant Names"
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]

# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

populator = _metadata.MetadataPopulator.with_model_file(TFLITE_MODEL)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files([LABEL_MODEL])
populator.populate()
# --- TensorFlow Lite Model Meta data --- #

# --- Model Metrics --- #
## 1. Training Accuracy and Loss
## 2. TFlite Confusion Matrix

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

fig.suptitle(MODEL_NAME)

fig.savefig(IMG_NAME, dpi=300, bbox_inches = "tight")

# Load Model Vars
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL) # load the file

interpreter.allocate_tensors() # get the tensor information

input_details = interpreter.get_input_details() # these are the details of what
    #  the input should look like
output_details = interpreter.get_output_details() # this is the details of what
    #  the output should look like
  
matrix = [[0 for x in range(len(classes))] for y in range(len(classes))]

hit = 0 # number of hits

# Global Vars
s = input_details[0]['shape'][1] # side length of the square that the images
    #  should be cropped to. we get the size that the input image is supposed
    #  to be according to the model file

# Run The Tests
for i in trange(len(test_gen), desc="Testing the Model"):
    #img = [0,test_gen[i]#.reshape(1, s, s, 3) # reshape the image to be 4D
    data = list(test_gen[i])[0][0]
    img = np.array(data).reshape(1,s,s,3)

    # Test model on the image
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke() # invoke the interpreter

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    j = int(np.argmax(list(test_gen[i])[1][0]))
    k = int(np.argmax(output_data))
    if j == k:
        hit = hit + 1

    matrix[j][k] += 1

# Create Graph
acc = "Accuracy: " + str(round(hit/len(test_gen)*100, 2)) + "%"

plt.imshow(matrix, 
          cmap=plt.cm.Blues, 
          vmax=df_test['class'].value_counts()[0]*1.35)
plt.title(MODEL_NAME+' Confusion Matrix\n'+acc)

for i in range(len(matrix)):
  for j in range(len(matrix[0])):
    plt.text(j, i, matrix[i][j], fontsize=12, ha="center", va="center")

plt.xticks(np.arange(0, len(classes)), classes, rotation=90)
plt.xlabel("Predicted Class")

plt.yticks(np.arange(0, len(classes)), classes)
plt.ylabel("Actual Class")

plt.grid(b=False)
plt.savefig(CONFUSION_NAME, dpi=300, bbox_inches = "tight")
# --- Model Metrics --- #
