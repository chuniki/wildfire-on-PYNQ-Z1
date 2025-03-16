import os
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
EXPERIMENTS_FOLDER = 'experiments/'
EXPERIMENTS_FOLDER += '750_FPS/'
if not os.path.isdir(EXPERIMENTS_FOLDER):
    os.mkdir(EXPERIMENTS_FOLDER)

RUN_FOLDER = '00_estimates_mvau_rtl/'

RUN_FOLDER = EXPERIMENTS_FOLDER + RUN_FOLDER
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    print(f'Run folder created in: {RUN_FOLDER}')

# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
NUM_CLASSES = 2

IMG_H = 224
IMG_W = 224
NUM_CHANNELS = 3

