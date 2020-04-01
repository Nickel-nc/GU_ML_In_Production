
# General settings:
# -----------------

TRAIN_DATA_FILEPATH, SEP = ('dataset/dataset_train.csv', ';')
RAW_TRAIN_PATH = 'train/'
RAW_TEST_PATH = 'test/'
DATASET_PATH = 'dataset/'  # folder for built dataset
OUTPUT_DATA_PATH = 'predicted_data.csv'

# Building raw data set settings:
# ------------------------------

# Set True to prepare data sets from scratch
REBUILD_TRAIN = False
REBUILD_TEST = False

# Choose a period of evaluation depending on input data range
CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

# Set evaluation intervals in days
INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

# Model Settings
# ---------------

MODEL_NAME = 'model.sav'
# Use model from file
LOAD_MODEL = False
# # Save changes in model
# SAVE_MODEL_CHANGES = True

USE_IMPORTANT_FEATURES = True
FEATURES_FILE = 'important_features.sav'




