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

# Visualise bounding box on image
def draw_box(image, box, color=[255,0,0]):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image

def parse_dataset(dataset_path):
    '''
    Parses dataset inside dataset_path folder and returns img data
    and x,y coordinates of phone
    Input: dataset_path: Path of dataset
    Returns:
        X: list of x coordinate (float) of phone for each image
        Y: list of y coordiates (float) of phone for each image
        imgs = Numpy array of images
        imgs_path: Meta data, contains file path of each image  
    '''
    filenames = [dataset_path + "/labels.txt"]
    dataset = tf.data.TextLineDataset(filenames)

    # Preparing training data
    imgs = []
    X = []
    Y = []
    Y_regression = []
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        for i in range(100):
            value = sess.run(next_element)
            vals = value.decode('utf-8').split(" ")
            imgs.append(vals[0])
            X.append(float(vals[1]))
            Y.append(float(vals[2]))

    # getting image data into numpy arrays
    img_paths = imgs.copy()
    for i in range(len(X)): 
        imgs[i] = cv2.imread(dataset_path + "/" + imgs[i])
    imgs = np.stack(imgs,axis=0)

    return X, Y, imgs, img_paths

X, Y, imgs, img_paths = parse_dataset('dataset')

# Defining Model and Dataset



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class PhoneConfig(Config):
    """Configuration for training on the objects of BDD dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bdd"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 1 for only traffic lights

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


class PhoneDataset(utils.Dataset):
    """Generates the dataset for phones for Mask RCNN
    """

    def load_shapes(self, mode):
    
        assert mode == "train" or  mode == "test" or mode == "val", "Mode can only be train, val or test"
        
        # Add classes
        
        self.add_class("phone", 1, "phone")
        
        self.class2index = {
            "phone": 1,
        }
        
        self.base_image_path = '/content/drive/My Drive/brainTest/dataset/'
        self.images = imgs.copy()
        self.img_paths = img_paths.copy()
        self.X = X.copy()
        self.Y = Y.copy()

        train_start = 0
        train_end = round(100*0.8 - 1)

        val_start = train_end+1
        val_end = round(val_start + 100*0.2 -1)

        # Load annotation file, define split
        if mode == "train":
          start = train_start
          end = train_end
        elif mode  == "val":
          start = val_start
          end = val_end
       
        # Add to database of base class
        for i in range(start,end+1):
          self.add_image("phone", i, self.img_paths[i])
        print(end-start, " examples")
    
    def load_image(self, image_id):
        return self.images[image_id]

    def image_reference(self, image_id):
        return ""
      
    def bounding_box(self, image_id):
        real_x = float(self.X[image_id])
        real_y = float(self.Y[image_id])
        bb = np.asarray([round((real_y - 0.02)*326), round((real_x - 0.02)*490), round((real_y + 0.02)*326), round((real_x + 0.02)*490) ])
        return self.draw_box(self.images[image_id].copy(),bb)
      
    def draw_box(self,image, box):
        """Draw 3-pixel width bounding boxes on the given image array.
        color: list of 3 int values for RGB.
        """
        x1, y1, x2, y2 = box
        color = (255,0,0)
        image[y1:y1 + 10, x1:x2] = color
        image[y2:y2 + 10, x1:x2] = color
        image[y1:y2, x1:x1 + 10] = color
        image[y1:y2, x2:x2 + 10] = color
        return image
    
    def load_mask(self, image_id):
        masks = []
        class_ids = []
       
        real_x = float(self.X[image_id])
        real_y = float(self.Y[image_id])
        bb = np.asarray([round((real_y - 0.02)*326), round((real_x - 0.02)*490), round((real_y + 0.02)*326), round((real_x + 0.02)*490) ])

        start = (bb[0],bb[1])
        extent = (bb[2],bb[3])
        mask = np.zeros([326, 490])
        rr, cc = draw.rectangle(start, extent,shape=[326,490])
        mask[rr, cc] = 1

        masks.append(mask.astype(np.bool))
        class_ids.append(self.class2index["phone"])
        masks = np.stack(masks, axis=2)
        class_ids = np.asarray(class_ids).astype(dtype=np.int32)
        return masks, class_ids

config = PhoneConfig()

dataset_train = PhoneDataset()
dataset_train.load_shapes("train")
dataset_train.prepare()

dataset_val = PhoneDataset()
dataset_val.load_shapes("val")
dataset_val.prepare()

# Load model
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Load coco model weights for object detection
model.load_weights(COCO_MODEL_PATH, by_name=True,
                exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                        "mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads')

# Ready model for inference