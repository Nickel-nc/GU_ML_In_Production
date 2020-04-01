import pickle

from Settings import *
from BuildDataset import build_dataset_raw
from ProcessData import *
from FitModel import xgb_fit, choose_important_features
from EvaluateModel import evaluation
from PredictData import get_predict


if REBUILD_TRAIN:

    """Сбор тренеровочного датасета"""

    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                      churned_end_date=CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=RAW_TRAIN_PATH,
                      dataset_path=DATASET_PATH,
                      mode='train')

if REBUILD_TEST:

    """Сбор тестового датасета"""

    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                      churned_end_date=CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=RAW_TEST_PATH,
                      dataset_path=DATASET_PATH,
                      mode='test')



dataset = pd.read_csv(TRAIN_DATA_FILEPATH, sep=SEP)

X, X_train_balanced, X_test, y_train_balanced, y_test = process_data(dataset)

if LOAD_MODEL:
    with open(MODEL_NAME, 'rb') as input_file:
        model = pickle.load(input_file)

if not LOAD_MODEL:
    # Creates new model
    model = xgb_fit(X_train_balanced, y_train_balanced, X_test, y_test)

if USE_IMPORTANT_FEATURES:
    try:
        with open(FEATURES_FILE, 'rb') as input_file:
            FEATURES = pickle.load(input_file)
            feats = 1
            print('Successfully loaded features')
    except IOError as e:
        print(u'Can\'t load features')
        feats = 0

    if feats == 0:
        FEATURES = choose_important_features(X, X_train_balanced, y_train_balanced, model)

    X_train_PI = pd.DataFrame(X_train_balanced, columns=X.columns)[FEATURES]
    X_test_PI = pd.DataFrame(X_test, columns=X.columns)[FEATURES]

    model = xgb_fit(X_train_PI, y_train_balanced, X_test_PI, y_test)

    with open(MODEL_NAME, 'wb') as f:
        pickle.dump(model, f)

predict_proba_test = model.predict_proba(X_test_PI)
predict_test = model.predict(X_test_PI)
precision, recall, f1, ll, roc_auc = evaluation(y_test, predict_test, predict_proba_test[:, 1])

get_predict(model, X_test_PI)

