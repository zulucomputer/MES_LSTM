#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '..')


import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from MES_LSTM.model import *
from utils.metrics import *

import warnings
warnings.simplefilter('ignore')

import argparse
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lstm_size", type = int, required = True, help = "size of lstm layer")
parser.add_argument('--epochs', type = int, required = True, help = 'number of training epochs')
# parser.add_argument('--alpha', type = float, required = False, help = 'significance level for prediction intervals')
args = parser.parse_args()


def setup_logging(run_path):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    log_dir_path = os.path.join(run_path, "logs")
    os.makedirs(log_dir_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir_path, f"run.log"), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

# logging
logger = setup_logging('.')

# log all parameters
logger.info("Command-line arguments")
for arg, value in sorted(vars(args).items()):
    logger.info(f"Argument {arg}: {value}")



pre_layer = preprocess(first_time = 0) # change first time to 1 if first time running to download data
df = pre_layer.load_data()
df = pre_layer.clean_data(df)
df = pre_layer.fill_missing(df)
scaled_df, df_scaler = pre_layer.scale(df)
# scaled_df

dl_layer = lstm(results_path = 'results/pure_lstm/', lstm_size = args.lstm_size, epochs = args.epochs)
train, valid, test, x_train, y_train, x_valid, y_valid, x_test = dl_layer.split(scaled_df)
y_pred_scaled = dl_layer.forecast_model(test, x_train, y_train, x_valid, y_valid, x_test)
forecasts = dl_layer.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)

pi_pred_scaled = dl_layer.pi_model(test, x_train, y_train, x_valid, y_valid, x_test)
pi = dl_layer.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)
# pi.head()


print(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
print(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
print(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
print(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))
print('============================================================================================')
print(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
print(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
print(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
print(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))



