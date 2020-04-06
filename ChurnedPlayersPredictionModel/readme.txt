  -=PROJECT DESCRIPTION=-

Predict is-churned online-gaming players based on XGBoost trained model
Uses local inputs and outputs

-- Preferences: "Settings.py"
-- Logging: "logs/logger.log"

Program options:#

##################
#### Option 1 ####
##################
#
# builds dataset from multiple .csv files from "raw_data/train", "raw_data/test" folders
# creates temp files "dataset_raw_train.csv", "dataset_raw_test.csv" in dataset folder
# dealing with missing values, outliers, categorical features
# creates files "dataset_train.csv", "dataset_test.csv" in dataset folder
#
# REQUIRES:
#
# -- sample.csv
# -- profiles.csv
# -- payments.csv
# -- reports.csv
# -- abusers.csv
# -- logins.csv
# -- pings.csv
# -- sessions.csv
# -- shop.csv
#
# OUTPUT:
#
# -- dataset_raw_test.csv
# -- dataset_raw_train.csv
# -- dataset_test.csv
# -- dataset_train.csv
#
#
##################
#### Option 2 ####
##################
#
# reads stored dataset
# makes train test split
# balance data with ADASYN (ratio=0.3)
# fit xgb classifier, optional: search model parameters
# calculate important features
# store intermediary results
#
# REQUIRES:
#
# -- dataset_raw_test.csv
# -- dataset_raw_train.csv
# -- dataset_test.csv
# -- dataset_train.csv
#
# OUTPUT:
#
# -- important_features.sav
# -- model.sav
# 
#
##################
#### Option 3 ####
##################
#
# read stored model, features, test data
# make and store predictions
#
# REQUIRES:
#
# -- important_features.sav
# -- model.sav
# -- dataset_test.csv
#
# OUTPUT:
#
# -- prediction.csv (fields: user_id, is_churned)
#
#
##################