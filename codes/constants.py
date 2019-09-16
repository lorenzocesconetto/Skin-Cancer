import os


################################
# Directories
################################
# Data Paths
DATA_ORIGINAL_DIR = 'data_original'
DATA_PROCESSED_DIR = 'data_processed'
DATA_PROCESSED_DIR_PATH = os.path.join('..', DATA_PROCESSED_DIR)
DATA_ORIGINAL_DIR_PATH = os.path.join('..', DATA_ORIGINAL_DIR)

# Splits
SPLIT_DIRS = ['train', 'test', 'validation']
TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR_PATH, SPLIT_DIRS[0])
TEST_PATH = os.path.join(DATA_PROCESSED_DIR_PATH, SPLIT_DIRS[1])
VAL_PATH = os.path.join(DATA_PROCESSED_DIR_PATH, SPLIT_DIRS[2])


################################
# Data
################################
# DIAGNOSTIC_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# NUMERIC_CLASSES = [4, 6, 2, 1, 0, 5, 3]
# NUMERIC_TO_NAME_CLASSES = {x: y for x, y in zip(NUMERIC_CLASSES, DIAGNOSTIC_CLASSES)}
# SORTED_NUMERIC_CLASSES, SORTED_DIAGNOSTIC_CLASSES = (list(t) for t in zip(*sorted(zip(NUMERIC_CLASSES, DIAGNOSTIC_CLASSES))))

DIAGNOSTIC_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
NUMERIC_CLASSES = [0, 1, 2, 3, 4, 5, 6]
DIAG_2_NUM = {x: y for x, y in zip(DIAGNOSTIC_CLASSES, NUMERIC_CLASSES)}
NUM_2_DIAG = {x: y for x, y in zip(NUMERIC_CLASSES, DIAGNOSTIC_CLASSES)}

# Image format
IMG_FORMAT = 'jpg'


