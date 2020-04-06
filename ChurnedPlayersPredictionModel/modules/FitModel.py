import xgboost as xgb
from utils.ELI5 import PermutationImportance
import pandas as pd
import pickle
from Settings import *
from modules.EvaluateModel import evaluation, search_params
from modules.ProcessData import process_data

def xgb_fit(X_train, y_train, parameters):

    clf = xgb.XGBClassifier(**parameters)
    print('Training XGB classification model...', end='')
    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    print('Done')
    return clf


def choose_important_features(X, X_train_balanced, y_train_balanced, fitted_clf):

    print('Calculating important features...', end='')
    perm = PermutationImportance(fitted_clf, random_state=42).fit(X_train_balanced, y_train_balanced)
    res = pd.DataFrame(X.columns, columns=['feature'])
    res['score'] = perm.feature_importances_
    res['std'] = perm.feature_importances_std_
    res = res.sort_values(by='score', ascending=False).reset_index(drop=True)
    good_features_PI = res.loc[res['score'] > 0]['feature']
    logger2.info(f"Calculated {good_features_PI.shape[0]} important features:")
    print(f'Selected {good_features_PI.shape[0]} important features ')
    # Saving calculated features
    with open(FEATURES_FILE, 'wb') as input_file:
        pickle.dump(good_features_PI, input_file)
        logger2.info(f"Features successfully saved to {FEATURES_FILE}")
    return good_features_PI


def train_model():

    dataset = pd.read_csv(TRAIN_DATA_FILEPATH, sep=SEP)  # Read stored data
    X, X_train_balanced, X_test, y_train_balanced, y_test = process_data(dataset)  # Split, balance data

    if USE_SEARCH_CV:
        try:
            with open(BEST_PARAMS_FILE, 'rb') as input_file:
                model_params = pickle.load(input_file)
            print(f"Load xgb.parameters from {BEST_PARAMS_FILE}")
            logger2.info(f"Load xgb.parameters from {BEST_PARAMS_FILE}")
        except FileNotFoundError:
            logger2.info("Perform RandomizedSearchCV for model parameters")
            model_params = search_params(X_train_balanced, y_train_balanced, MODEL_SEARCH_PARAMS)
        except:
            print("ERROR. Process aborted during evaluation model parameters")

    if not USE_SEARCH_CV:
        model_params = MODEL_BASE_PARAMS
        print(f"No Search CV. Base model parameters used")
        logger2.info(f"No Search CV. Base model parameters used")

    model = xgb_fit(X_train_balanced, y_train_balanced, model_params)  # Creates new model

    if TRAIN_IMPORTANT_FEATURES:
        FEATURES = choose_important_features(X, X_train_balanced, y_train_balanced, model)

    if not TRAIN_IMPORTANT_FEATURES:
        try:
            with open(FEATURES_FILE, 'rb') as input_file:
                FEATURES = pickle.load(input_file)
            print(f"Loading Important Features from backup: {len(FEATURES)} features")
            logger2.info(f"Load Important Features from backup: {len(FEATURES)} features")
        except FileNotFoundError as e:
            logger2.error(e)

    X_train_PI = pd.DataFrame(X_train_balanced, columns=X.columns)[FEATURES]
    X_test_PI = pd.DataFrame(X_test, columns=X.columns)[FEATURES]
    model = xgb_fit(X_train_PI, y_train_balanced, model_params)
    logger2.info(f'Trained model parameters:\n{model}')
    predict_proba_test = model.predict_proba(X_test_PI)
    predict_test = model.predict(X_test_PI)
    evaluation(y_test, predict_test, predict_proba_test[:, 1])

    with open(MODEL_PATH, 'wb') as input_file:
        pickle.dump(model, input_file)
        logger2.info(f"Model successfully saved to {MODEL_PATH}")
