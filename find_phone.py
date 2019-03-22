# Imports
import glob
import os
import numpy as np 
import cv2
import sys
import random
import math
import re
import time
import matplotlib
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import skimage.draw as draw

# Import Mask RCNN
ROOT_DIR = os.path.relpath('Mask_RCNN')

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Pointing to directory where model weights will be saved
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Helper functions for getting inference results
def returnCenterPoint(bbox):
    center_x = bbox[0] + (bbox[2] - bbox[0])/2
    center_y = bbox[1] + (bbox[3] - bbox[1])/2
    return center_x/512, (center_y-85)/340.63

def predict(r):
    y1,x1,y2,x2 = r['rois'][0]
    return returnCenterPoint([x1,y1,x2,y2])

# Setting Up Config class for Phone Dataset
class PhoneConfig(Config):
    """
        Configuration for training on the objects of Phone dataset.
        Derives from the base Config class and overrides values specific
        to the phone dataset.
    """
    # Give the configuration a recognizable name
    NAME = "phone"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 1 for for phone

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

# Setting up config for inference of trained model
class InferenceConfig(PhoneConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
model.load_weights(model_path, by_name=True)

# Helper funtion to load image and resize it for input
# into model. The model normalises the image by subtracting
# the mean channels of the images in the image net dataset internally
def load_image(file_path, config):
    image = cv2.imread(file_path)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    return image

# Get command line path for file path
file_path = sys.argv[1] 
original_image = load_image(file_path,config=inference_config)

# Perform inference
results = model.detect([original_image], verbose=0)
r = results[0] # take 1'st result

# Get center x,y coords of predicted bounding box
x,y = predict(r)
# Print predicted x, y coordnates
print(x,y)
