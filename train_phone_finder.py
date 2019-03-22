# The training pipeline of this phone detector utilises a open source 
# implementation of the Mask RCNN network. I am referring to the famous 
# Matterport implementation. I have parsed the inputs by turning images into 
# numpy arrays and getting the bounding box coordinates of the x,y labels
# that were given in labels.txt

# I converted the x,y coordiates into bounding box coords by taking x,y coordimates
# as the center of the boounding box. Then I cropped a bounding box of
# size 0.02 times of the image height and width.

# I experimented with different values but this gave the best accuracy.

# After getting the bounding box coordiates, I prepared a Dataset class that 
# was compatible with the Mask RCNN and trained the dataset for 5 epochs.
# I used a training and validation split of 80% and 20% of the dataset.

# I was able to train successfully and get good results.

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

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

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

# Setting up Dataset for Phones 
class PhoneDataset(utils.Dataset):
    """Generates the dataset for phones for Mask RCNN
    """

    def load_shapes(self, mode, X, Y, imgs, img_paths):
    
        assert mode == "train" or  mode == "test" or mode == "val", "Mode can only be train, val or test"
        
        # Add classes
        self.add_class("phone", 1, "phone")
        self.class2index = {
            "phone": 1,
        }
        
        # Assigning class variables
        self.images = imgs
        self.img_paths = img_paths
        self.X = X
        self.Y = Y
        self.img_h = 326 # all images are 326 pixels in height
        self.img_w = 490  # all images are 326 pixels in width
        self.bboxSize = 0.02 # as described above 0.02 is a good bounding box size

        # Dividing dataset into training and validation set.
        train_start = 0
        train_end = round(100*0.8 - 1) # 80% split

        val_start = train_end+1
        val_end = round(val_start + 100*0.2 -1) # 20% split

        # Load annotation file, define split
        if mode == "train":
          start = train_start
          end = train_end
        elif mode  == "val":
          start = val_start
          end = val_end
       
        # Adding to database of base class (Mask RCNN requirement)
        for i in range(start,end+1):
          self.add_image("phone", i, self.img_paths[i])
        print(end-start, " examples")
    
    def load_image(self, image_id):
        # Returns numpy array of image
        return self.images[image_id]

    def image_reference(self, image_id):
        # Not needed
        return ""  
    
    def load_mask(self, image_id):
        '''
        This function returns a list of segmentation masks of the objects
        to be recognised in the image. The Matterpot Mask RCNN takes only
        segmentation masks as input into it. So, we have to convert our
        bounding boxes to segmentation mask. Luckily, skimage provides 
        several functions that make this process easy.
        '''

        masks = []
        class_ids = []
       
        real_x = float(self.X[image_id]) # x coordinate ground truth
        real_y = float(self.Y[image_id]) # y coordinate ground truth

        # Build bounding box
        bbox = np.asarray([round((real_y - self.bboxSize)*self.img_h),
         round((real_x - self.bboxSize)*self.img_w),
          round((real_y +self.bboxSize)*self.img_h),
           round((real_x + self.bboxSize)*self.img_w) ])

        # Build segmentation mask
        start = (bbox[0],bbox[1])
        extent = (bbox[2],bbox[3])
        mask = np.zeros([326, 490])
        rr, cc = draw.rectangle(start, extent,shape=[326,490])
        mask[rr, cc] = 1

        # Return segmentation mask and class id of each mask
        masks.append(mask.astype(np.bool))
        class_ids.append(self.class2index["phone"])
        masks = np.stack(masks, axis=2)
        class_ids = np.asarray(class_ids).astype(dtype=np.int32)

        return masks, class_ids


# Parse dataset into images and x,y coordiates
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

# Prepare Dataset for training and validation
X, Y, imgs, img_paths = parse_dataset('dataset')

config = PhoneConfig()

dataset_train = PhoneDataset()
dataset_train.load_shapes("train", X, Y, imgs, img_paths)
dataset_train.prepare()

dataset_val = PhoneDataset()
dataset_val.load_shapes("val", X, Y, imgs, img_paths)
dataset_val.prepare()

# Prepare model for training

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

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

# Model is saved every epoch in MODEL_DIR
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads')
