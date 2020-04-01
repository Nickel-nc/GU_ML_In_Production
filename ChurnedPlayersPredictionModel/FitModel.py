import xgboost as xgb
from ELI5 import PermutationImportance
import pandas as pd
import pickle
from Settings import *
from EvaluateModel import evaluation

def xgb_fit(X_train, y_train, X_test, y_test):

    print('Training XGB classification model...', end='')
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight=3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

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
    print('Selected features:', good_features_PI.shape[0])

    # Saving calculated features
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(good_features_PI, f)

    return good_features_PI