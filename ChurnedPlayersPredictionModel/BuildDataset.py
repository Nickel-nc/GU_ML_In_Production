import pandas as pd
import time
from datetime import datetime, timedelta
from Settings import *


def time_format(sec):
    return str(timedelta(seconds=sec))


def build_dataset_raw(inter_list: object,
                      raw_data_path: object,
                      dataset_path: object,
                      mode: object):
    print(f'Start building {mode} dataset...')

    start_t = time.time()

    sample = pd.read_csv('{}sample.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    profiles = pd.read_csv('{}profiles.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    payments = pd.read_csv('{}payments.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    reports = pd.read_csv('{}reports.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    abusers = pd.read_csv('{}abusers.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    logins = pd.read_csv('{}logins.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    pings = pd.read_csv('{}pings.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    sessions = pd.read_csv('{}sessions.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    shop = pd.read_csv('{}shop.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')

    print('Run time (reading csv files): {}'.format(time_format(time.time() - start_t)))
    logger2.info('Run time (reading csv files): {}'.format(time_format(time.time() - start_t)))
    # -----------------------------------------------------------------------------------------------------
    print('NO dealing with outliers, missing values and categorical features...')
    # -----------------------------------------------------------------------------------------------------
    # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

    print('Creating dataset...')
    logger2.info(f'Creating {mode} dataset')
    # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
    if mode == 'train':
        dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
    elif mode == 'test':
        dataset = sample.copy()[['user_id', 'level', 'donate_total']]

    # Пройдемся по всем источникам, содержащим "динамичекие" данные
    for df in [payments, reports, abusers, logins, pings, sessions, shop]:

        # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
        data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
        data['day_num_before_churn'] = 1 + (data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                                            data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(
            lambda x: x.days)
        df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

        # Для каждого признака создадим признаки для каждого из времененно интервала (4 интервала по 7 дней)
        features = list(set(data.columns) - set(['user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn']))
        print('Processing with features:', features)
        logger2.info(f'Processing with features: {features}')
        for feature in features:
            for i, inter in enumerate(inter_list):
                inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)]. \
                    groupby('user_id')[feature].mean().reset_index(). \
                    rename(index=str, columns={feature: feature + '_{}'.format(i + 1)})
                df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

        # Добавляем построенные признаки в датасет
        dataset = pd.merge(dataset, df_features, how='left', on='user_id')

        print(f'Run time (calculating features): {time_format(time.time() - start_t)}')
        logger2.info(f'Run time (calculating features): {time_format(time.time() - start_t)}')

    # Добавляем "статические" признаки
    dataset = pd.merge(dataset, profiles, on='user_id')
    # ---------------------------------------------------------------------------------------------------------------------------
    dataset.to_csv('{}dataset_raw_{}.csv'.format(dataset_path, mode), sep=';', index=False)
    print('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
          format(dataset_path, time_format(time.time() - start_t)))
    logger2.info('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
                 format(dataset_path, time_format(time.time() - start_t)))


def prepare_dataset(data_filepath,
                    dataset_type,
                    dataset_path):
    dataset = pd.read_csv(data_filepath, sep=SEP)

    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')
    logger2.info('Dealing with missing values, outliers, categorical features')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1, len(INTER_LIST) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=SEP, index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
          format(dataset_path, time_format(time.time() - start_t)))

    logger2.info('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
                 format(dataset_path, time_format(time.time() - start_t)))
