import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    NUM_CATEGORY = 1+18 #(background+classes)
    NUM_GROUP = 100
    NUM_POINT = 18000 #18000
    # Num of points for each generated instance
    NUM_POINT_INS = 512
    BATCH_SIZE = 2
    # How many seed points to sample for generation
    NUM_SAMPLE = 256 # training - 256, test - 2048 

    # ROIs kept after sorting and before non-maximum suppression
    SPN_PRE_NMS_LIMIT = 192 # training - 192, test - 1536
    # ROIs kept after non-maximum suppression (training and inference)
    SPN_NMS_MAX_SIZE_TRAINING = 128 # training - 128, test - 512
    SPN_NMS_MAX_SIZE_INFERENCE = 96 # training - 96, test - 384 
    SPN_IOU_THRESHOLD = 0.5
    SPN_SCORE_THRESHOLD = float('-inf')

    NUM_POINT_INS_MASK = 256 # training - 256, test - 1024
    TRAIN_ROIS_PER_IMAGE = 64 # training - 64, test - 512 
    ROI_POSITIVE_RATIO = 0.33
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    NORMALIZE_CROP_REGION = True

    SHRINK_BOX = False # Default: False
    USE_COLOR = True
    TRAIN_MODULE = ['SPN'] # Option: ['SPN', 'RPOINTNET']

    DETECTION_MIN_CONFIDENCE = 0.7 # Default: 0.7
    DETECTION_NMS_THRESHOLD = 0.1 # Default: 0.1
    DETECTION_MAX_INSTANCES = 100


    def __init__(self, istrain=True):
        """Set values of computed attributes."""
        if not istrain:
            NUM_SAMPLE = 2048 # training - 256, test - 2048 
            # ROIs kept after sorting and before non-maximum suppression
            SPN_PRE_NMS_LIMIT = 1536 # training - 192, test - 1536
            # ROIs kept after non-maximum suppression (training and inference)
            SPN_NMS_MAX_SIZE_TRAINING = 512 # training - 128, test - 512
            SPN_NMS_MAX_SIZE_INFERENCE = 384 # training - 96, test - 384 
            NUM_POINT_INS_MASK = 1024 # training - 256, test - 1024
            TRAIN_ROIS_PER_IMAGE = 512 # training - 64, test - 512 
            DETECTION_MIN_CONFIDENCE = 0.5


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")