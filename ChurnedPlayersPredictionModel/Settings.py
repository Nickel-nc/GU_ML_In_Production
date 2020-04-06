import logging.config
import logging
from utils.LoggerSetup import dictLogConfig

# General settings:
# -----------------

README_FILE = 'readme.txt'

# Logger settings
logger1 = logging.getLogger('runSession')
logger2 = logging.getLogger('progress')
logging.config.dictConfig(dictLogConfig)

# Buildingdata set settings:
# ------------------------------

TRAIN_DATA_FILEPATH, SEP = ('dataset/dataset_train.csv', ';')
TEST_DATA_FILEPATH = 'dataset/dataset_test.csv'
RAW_TRAIN_PATH: str = 'raw_data/train/'
RAW_TEST_PATH: str = 'raw_data/test/'
DATASET_PATH: str = 'dataset/'  # folder for built dataset
OUTPUT_DATA_PATH: str = 'output/YN_prediction.csv'

# Choose a period of evaluation depending on input data range
CHURNED_START_DATE: str = '2019-09-01'
CHURNED_END_DATE: str = '2019-10-01'

# Set evaluation intervals in days
INTER_1: tuple = (1, 7)
INTER_2: tuple = (8, 14)
INTER_3: tuple = (15, 21)
INTER_4: tuple = (22, 28)
INTER_LIST: list = [INTER_1, INTER_2, INTER_3, INTER_4]

# Model Settings
# ---------------

MODEL_PATH = 'output/model.sav'
FEATURES_FILE = 'output/important_features.sav'
BEST_PARAMS_FILE = 'output/xgb_best_params.sav'
TRAIN_IMPORTANT_FEATURES = False  # if False, get feature list from saved file. Otherwise calculate
USE_SEARCH_CV = False  # if False, get params list from saved file or base parameters. Otherwise calculate
MODEL_BASE_PARAMS = {'max_depth': 3,
                       'n_estimators': 100,
                       'learning_rate': 0.1,
                       'subsample': 1,
                       'colsample_bytree': 0.5,
                       'min_child_weight': 3,
                       'reg_alpha': 0,
                       'reg_lambda': 0,
                       'seed': 42,
                       'missing': 1e10
                       }

MODEL_SEARCH_PARAMS = {'max_depth':[3, 5, 7],
                       'n_estimators':[50, 100, 150, 200],
                       'learning_rate':[ 0.01, 0.05, 0.1, 0.5],
                       'subsample':[0.5, 1],
                       'colsample_bytree': [0.5, 0.7, 1],
                       'min_child_weight': [3, 5, 7],
                       'reg_alpha': [0.01, 0],
                       'reg_lambda': [0],
                       'seed': [42],
                       'missing': [1e10]
                       }
