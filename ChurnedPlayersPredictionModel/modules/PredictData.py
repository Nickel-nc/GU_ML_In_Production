from Settings import *
import pandas as pd

def get_predict(model, test_data_filepath, feats):

    logger2.info(f'Start prediction...')
    data = pd.read_csv(test_data_filepath, sep=SEP)
    x_test = data[feats]
    y_pred = model.predict(x_test)
    data['is_churned'] = y_pred
    data[['user_id', 'is_churned']].to_csv(OUTPUT_DATA_PATH, index=None)
    logger2.info(f'Prediction succesfully saved to {OUTPUT_DATA_PATH}')
    print(f'Prediction succesfully saved to {OUTPUT_DATA_PATH}')
