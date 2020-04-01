from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(dataset):

    print('Splitting data...', end='')

    X = dataset.drop(['user_id', 'is_churned'], axis=1)
    y = dataset['is_churned']

    X_mm = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=100)
    print('Done')
    return X, X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):

    print('Balancing minor class...', end='')

    # Balancing minor class
    X_train_balanced, y_train_balanced = SMOTE(sampling_strategy=0.3, random_state=42).fit_sample(X_train, y_train)

    print('Done')

    return X_train_balanced, y_train_balanced


def process_data(df):
    X, X_train, X_test, y_train, y_test = split_data(df)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    return X, X_train_balanced, X_test, y_train_balanced, y_test





