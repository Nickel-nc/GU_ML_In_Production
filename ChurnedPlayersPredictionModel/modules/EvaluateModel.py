from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, KFold
import xgboost as xgb
import pickle
from Settings import *

def evaluation(y_true, y_pred, y_prob):
    print('Evaluating model parameters...')
    logger2.info('Evaluating model parameters')
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('Log Loss: {}'.format(ll))
    print('ROC AUC: {}'.format(roc_auc))

    logger2.info(f'Precision: {precision}\nRecall: {recall}\nF1: {f1}\nLog Loss: {ll}\nROC AUC: {roc_auc}')
    return precision, recall, f1, ll, roc_auc

def search_params(X_train, y_train, params):
    print('Searching best model parameters...', end='')
    cv = KFold(n_splits=3, random_state=42, shuffle=True)
    rs = RandomizedSearchCV(xgb.XGBClassifier(), params, scoring='f1', cv=cv, n_jobs=-1)
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    with open(BEST_PARAMS_FILE, 'wb') as f:
        pickle.dump(best_params, f)
    logger2.info(f"CV Parameters result:\n{best_params}")
    logger2.info(f"CV Parameters successfully saved to {BEST_PARAMS_FILE}")
    print('Done')

    return best_params

