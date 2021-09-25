#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '..')


import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MES_LSTM.model import *
from utils.metrics import *

import warnings
warnings.simplefilter('ignore')

import argparse
import logging
import os
from numpy import arange

# parser = argparse.ArgumentParser()
# parser.add_argument("--lstm_size", type = int, required = True, help = "size of lstm layer")
# parser.add_argument('--epochs', type = int, required = True, help = 'number of training epochs')
# # parser.add_argument('--alpha', type = float, required = False, help = 'significance level for prediction intervals')
# args = parser.parse_args()


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

# # log all parameters
# logger.info("Command-line arguments")
# for arg, value in sorted(vars(args).items()):
#     logger.info(f"Argument {arg}: {value}")


lstm_sizes = arange(50, 155, 5)
epoch_sizes = arange(15, 85, 10)

rmse_deaths = pd.DataFrame(np.zeros((len(lstm_sizes), len(epoch_sizes)), dtype=float))
smape_deaths = pd.DataFrame(np.zeros((len(lstm_sizes), len(epoch_sizes)), dtype=float)) #
rmse_cases = pd.DataFrame(np.zeros((len(lstm_sizes), len(epoch_sizes)), dtype=float))
smape_cases = pd.DataFrame(np.zeros((len(lstm_sizes), len(epoch_sizes)), dtype=float)) #

warnings.simplefilter('ignore')

pre_layer = preprocess(first_time = 0) # change first time to 1 if first time running to download data
df = pre_layer.load_data()
df = pre_layer.clean_data(df)
df = pre_layer.fill_missing(df)
scaled_df, df_scaler = pre_layer.scale(df)

for p in range(len(lstm_sizes)):
#     tf.keras.backend.clear_session()
    for q in range(len(epoch_sizes)):
#         tf.keras.backend.clear_session()
        # run lstm

        dl_layer = lstm(results_path = 'results/pure_lstm/', lstm_size = lstm_sizes[p], epochs = epoch_sizes[q])
        train, valid, test, x_train, y_train, x_valid, y_valid, x_test = dl_layer.split(scaled_df)
        y_pred_scaled = dl_layer.forecast_model(test, x_train, y_train, x_valid, y_valid, x_test)
        forecasts = dl_layer.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)

#         pi_pred_scaled = dl_layer.pi_model(test, x_train, y_train, x_valid, y_valid, x_test)
#         pi = dl_layer.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)
                           
        smape_deaths.iloc[p, q] = smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred'])
        smape_cases.iloc[p, q] = smape(forecasts['total_cases_true'], forecasts['total_cases_pred'])
        rmse_deaths.iloc[p, q] = rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred'])
        rmse_cases.iloc[p, q] = rmse(forecasts['total_cases_true'], forecasts['total_cases_pred'])
        
        print('---------------------------------------------------------------------------------------------')
        print('[INFO] running model {} out of {}'.format((p + 1) * (q + 1), len(lstm_sizes) * len(epoch_sizes)))
#         print(rmse_deaths.tail())
            
            
# pi.head()

tab_path = 'tables/'
os.makedirs(tab_path, exist_ok = True)
smape_deaths.to_excel(tab_path + 'smape_deaths.xlsx')
smape_cases.to_excel(tab_path + 'smape_cases.xlsx')
rmse_deaths.to_excel(tab_path + 'rmse_deaths.xlsx')
rmse_cases.to_excel(tab_path + 'rmse_cases.xlsx')

# print(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
# print(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
# print(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
# print(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))

print('============================================= * DONE * ===============================================')



