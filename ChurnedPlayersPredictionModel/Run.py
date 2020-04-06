import warnings
warnings.filterwarnings('ignore')
import pickle
from Settings import *
from modules.BuildDataset import build_dataset_raw, prepare_dataset
from modules.FitModel import train_model
from modules.PredictData import get_predict
logger1.info(f"{'#' * 20} Session started {'#' * 20}")

while True:
    # Prints console interface
    print('\nMenu:')
    print(' 1 - Build Dataset')
    print(' 2 - Train Model')
    print(' 3 - Make Predictions')
    print(' 4 - Show Program Description')
    print(' Press [enter] to Exit Program')
    try:
        i = int(input('Choose the option: ') or 0)
    except:
        continue
    finally:
        print()

    if not i:  # Exit program
        logger1.info(f"{'#' * 20} Session closed {'#' * 20}")
        print('Closing app...')
        break

    if i == 1:  # Build dataset
        logger2.info(f"____OPTION 1_____ Start build dataset")
        try:
            build_dataset_raw(inter_list=INTER_LIST,
                              raw_data_path=RAW_TRAIN_PATH,
                              dataset_path=DATASET_PATH,
                              mode='train')
            build_dataset_raw(inter_list=INTER_LIST,
                              raw_data_path=RAW_TEST_PATH,
                              dataset_path=DATASET_PATH,
                              mode='test')
            prepare_dataset(data_filepath=TRAIN_DATA_FILEPATH,
                            dataset_type='train',
                            dataset_path=DATASET_PATH)
            prepare_dataset(data_filepath=TEST_DATA_FILEPATH,
                            dataset_type='test',
                            dataset_path=DATASET_PATH)
        except FileNotFoundError as e:
            logger2.error(e)
            print('Data files not exists')
        except:
            print('Error occurred while building raw dataset')

    if i == 2:  # Train model
        logger2.info(f"____OPTION 2_____ Start Train model")
        try:
            train_model()  # use stored dataset in TRAIN_DATA_FILEPATH
        except FileNotFoundError as e:
            logger2.error(e)
            print('Data file not exists')
        except:
            print('Error occurred while training dataset')

    if i == 3:  # Make predictions
        logger2.info(f"____OPTION 3_____ Start Make predictions")
        try:
            with open(MODEL_PATH, 'rb') as input_file:
                model = pickle.load(input_file)            # Load model
            # with open(X_TEST_PATH, 'rb') as input_file:
            #     X_test_PI = pickle.load(input_file)  # Load stored test data

            with open(FEATURES_FILE, 'rb') as input_file:
                FEATURES = pickle.load(input_file)
            get_predict(model, TEST_DATA_FILEPATH, FEATURES)
        except FileNotFoundError as e:
            logger2.error(e)
            print('Required saved model and X_test data')
        except:
            logger2.error("Error during prediction")
            print('Error during prediction')

    if i == 4:
        try:
            with open(README_FILE, 'r') as f:
                for line in f:
                    print(line.strip())
        except FileNotFoundError as e:
            logger2.error(e)
            print('Readme not found')

