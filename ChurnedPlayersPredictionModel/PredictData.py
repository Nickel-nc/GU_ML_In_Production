import pandas as pd
from Settings import *

def get_predict(model, X_test):

    y_pred = model.predict(X_test)
    X_test['is_churned'] = y_pred
    X_test.to_csv(OUTPUT_DATA_PATH)
    print(f'Prediction succesfully saved to {OUTPUT_DATA_PATH}')
