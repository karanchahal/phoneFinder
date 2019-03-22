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
import tensorflow as tf
import skimage.draw as draw

ROOT_DIR = os.path.relpath('Mask_RCNN')
# # Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# For inference results
def returnCenterPoint(bbox):
    center_x = bbox[0] + (bbox[2] - bbox[0])/2
    center_y = bbox[1] + (bbox[3] - bbox[1])/2
    return center_x/512, (center_y-85)/340.63

def predict(r):
    y1,x1,y2,x2 = r['rois'][0]
    return returnCenterPoint([x1,y1,x2,y2])


class InferenceConfig(BDDConfig):
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
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

    
# TODO
file_path = ''
original_image = cv2.imread(file_path)

results = model.detect([original_image], verbose=1)
r = results[0] # take 1st result

print(predict(r))