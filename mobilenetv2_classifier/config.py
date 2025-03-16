import os
import torch

# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
RUN_FOLDER = 'experiments/' + '23_mini_resnet_70k_full_ds/'
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
LOGS_FOLDER = RUN_FOLDER + 'logs/'
if not os.path.isdir(LOGS_FOLDER):
    os.mkdir(LOGS_FOLDER)
PLOTS_FOLDER = RUN_FOLDER + 'plots/'
if not os.path.isdir(PLOTS_FOLDER):
    os.mkdir(PLOTS_FOLDER)
WEIGHTS_FOLDER = RUN_FOLDER + 'weights/'
if not os.path.isdir(WEIGHTS_FOLDER):
    os.mkdir(WEIGHTS_FOLDER)
ONNX_FOLDER = RUN_FOLDER + 'onnx/'
if not os.path.isdir(ONNX_FOLDER):
    os.mkdir(ONNX_FOLDER)
# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
NUM_CLASSES = len(CLASSES)

#___   Padding Model  ___#
IMG_DIM = {'W':224, 'H':224} # (W, H)

IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']
NUM_CHANNELS = 3

# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #
dfire_dir = '../../datasets/ds2fire/dfire_yolo/'
DFIRE_TRAIN_IMG_DIR = dfire_dir + 'train/images/'
DFIRE_TRAIN_LABEL_DIR = dfire_dir + 'train/labels/'
DFIRE_TEST_IMG_DIR = dfire_dir + 'test/images/'
DFIRE_TEST_LABEL_DIR = dfire_dir + 'test/labels/'

FASDD_UAV_IMGS_DIR = '../../datasets/fasdd/fasdd_uav/images/'
FASDD_UAV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/train.txt'
FASDD_UAV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/val.txt'
FASDD_UAV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/test.txt'

FASDD_CV_IMGS_DIR = '../../datasets/fasdd/fasdd_cv/images/'
FASDD_CV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/train.txt'
FASDD_CV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/val.txt'
FASDD_CV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/test.txt'

DS_LEN = None


# ______________________________________________________________________ #
#                   Hyperparameters and More                             #
# ______________________________________________________________________ #
BREVITAS_MODEL = True
MODEL = "Mobilenetv2_Mini_Resnet"
WIDTH_MULT = 1.0

LEARNING_RATE = 1e-3
# Optimizer
WEIGHT_DECAY = 0.0
FACTOR = 0.8
PATIENCE = 2
THRES = 0.001
MIN_LR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 150

LOAD_MODEL = False
LOAD_MODEL_DIR = './experiments/23_mini_resnet_70k_full_ds/weights/'
LOAD_MODEL_FILE = LOAD_MODEL_DIR + "Mobilenetv2_Mini_Resnet_classifier__best_mean_F1.pt"


LOSS_FN = "BCE"
SMOKE_PRECISION_WEIGHT = 0.8

# ______________________________________________________________________ #
#                             Quantization                               #
# ______________________________________________________________________ #
### Maybe not used, if model is loaded with default parameters
WEIGHTS_BIT_WIDTH = 4
ACTIVATIONS_BIT_WIDTH = 4
BIAS_BIT_WIDTH = 4

