#!/usr/bin/env python
# coding: utf-8


import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from os.path import join
from os import makedirs
from MES_LSTM.model import *
from utils.metrics import *
import argparse
import logging
import os

import warnings
warnings.simplefilter('ignore')

# check version
print(tf.keras.__version__)
print(tf.__version__)
print(tfp.__version__)
print(tf.config.list_physical_devices('GPU'))


parser = argparse.ArgumentParser()
parser.add_argument("--country", type = str, required = True, help = "which country to execute model on")
parser.add_argument('--thresh', type = float, required = False, help = 'threshold of missing data if column is to be deleted')
parser.add_argument('--alpha', type = float, required = False, help = 'significance level for prediction intervals')
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



runs = 36                          

# MES_RNN model

# pre-processing layer
if args.thresh:
    pre_layer = preprocess(first_time = 0, loc = args.country.replace('_', ' '), thresh = args.thresh) # change first_time to 1 if first time running to download data
else:
    pre_layer = preprocess(first_time = 0, loc = args.country.replace('_', ' ')) # use default thresh
df = pre_layer.load_data()
df = pre_layer.clean_data(df)
df = pre_layer.fill_missing(df)
scaled_df, df_scaler = pre_layer.scale(df)
scaled_df

# exponential smoothing layer
mes_layer = ES(loc = args.country, alpha = args.alpha)
params, internals = mes_layer.es(scaled_df)
es_scaled, df_trend, df_seas = mes_layer.deTS(scaled_df, internals)

es_scaled

results_deaths = pd.DataFrame(columns = ['smape_meslstm', 'rmse_meslstm', 'mis_meslstm', 'cov_meslstm',
                                     'smape_lstm', 'rmse_lstm', 'mis_lstm', 'cov_lstm',
                                     'smape_varmax', 'rmse_varmax', 'mis_varmax', 'cov_varmax',
                                     'smape_sarimax', 'rmse_sarimax', 'mis_sarimax', 'cov_sarimax',
                                     'smape_mlr', 'rmse_mlr', 'mis_mlr', 'cov_mlr'])
results_cases = pd.DataFrame(columns = ['smape_meslstm', 'rmse_meslstm', 'mis_meslstm', 'cov_meslstm',
                                     'smape_lstm', 'rmse_lstm', 'mis_lstm', 'cov_lstm',
                                     'smape_varmax', 'rmse_varmax', 'mis_varmax', 'cov_varmax',
                                     'smape_sarimax', 'rmse_sarimax', 'mis_sarimax', 'cov_sarimax',
                                     'smape_mlr', 'rmse_mlr', 'mis_mlr', 'cov_mlr'])
for run in range(runs):
    
    print(f'[INFO] processing for: {args.country}')
    print(f'[INFO] trial number: {str(run)} for country: {args.country}') 

    # deep learning layer
    dl_layer = lstm(loc = args.country, alpha = args.alpha)
    train, valid, test, x_train, y_train, x_valid, y_valid, x_test = dl_layer.split(es_scaled)
    y_pred_es_scaled = dl_layer.forecast_model(test, x_train, y_train, x_valid, y_valid, x_test)
    forecasts = dl_layer.reTS(y_pred_es_scaled, es_scaled, train, valid, df_trend, df_seas, df_scaler, df)

    # prediction intervals
    pi_pred_es_scaled = dl_layer.pi_model(test, x_train, y_train, x_valid, y_valid, x_test)
    pi = dl_layer.reTS_pi(pi_pred_es_scaled, es_scaled, train, valid, df_trend, df_seas, df_scaler, df)


    # metrics
    result_deaths = []
    result_cases = []
    result_deaths.append(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
    result_cases.append(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
    result_deaths.append(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
    result_cases.append(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))

#     tf.keras.backend.clear_session()


    # LSTM

    dl_layer = lstm(results_path = 'results/pure_lstm/',
                    loc = args.country,
                    alpha = args.alpha,
                    lstm_size = 50,
                    epochs = 25,
                    n_input_train = 14,
                    b_size_train = 16,
                    n_input_valid = 7,
                    b_size_valid = 16)
    train, valid, test, x_train, y_train, x_valid, y_valid, x_test = dl_layer.split(scaled_df)
    y_pred_scaled = dl_layer.forecast_model(test, x_train, y_train, x_valid, y_valid, x_test)
    forecasts = dl_layer.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)

    # prediction intervals
    pi_pred_scaled = dl_layer.pi_model(test, x_train, y_train, x_valid, y_valid, x_test)
    pi = dl_layer.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)

    # metrics
    result_deaths.append(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
    result_cases.append(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
    result_deaths.append(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
    result_cases.append(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))


#     tf.keras.backend.clear_session()

    #-----

    # VARMAX

    bench = stats(loc = args.country, alpha = args.alpha)
    train, test, x_train, x_test = bench.split(scaled_df)
    y_pred_scaled, pi_pred_scaled = bench.forecast_varmax(test, x_train, y_train, x_test)
    forecasts = bench.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)
    pi = bench.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)


    result_deaths.append(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
    result_cases.append(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
    result_deaths.append(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
    result_cases.append(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))

    # SARIMAX

    bench = stats(loc = args.country, results_path = 'results/sarimax/', alpha = args.alpha)
    # train, test, x_train, x_test = bench.split(scaled_df)
    y_pred_scaled, pi_pred_scaled = bench.forecast_sarimax(test, x_train, y_train, x_test)
    forecasts = bench.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)
    pi = bench.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)


    result_deaths.append(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
    result_cases.append(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
    result_deaths.append(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
    result_cases.append(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))


    # MLR

    bench = stats(loc = args.country, results_path = 'results/mlr/', alpha = args.alpha)
    # train, test, x_train, x_test = bench.split(scaled_df)
    y_pred_scaled, pi_pred_scaled = bench.forecast_mlr(test, x_train, y_train, x_test)
    forecasts = bench.descale(y_pred_scaled, scaled_df, train, valid, df_scaler, df)
    pi = bench.descale_pi(pi_pred_scaled, scaled_df, train, valid, df_scaler, df)

    result_deaths.append(smape(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(smape(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(rmse(forecasts['total_deaths_true'], forecasts['total_deaths_pred']))
    result_cases.append(rmse(forecasts['total_cases_true'], forecasts['total_cases_pred']))
    result_deaths.append(mis(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values, alpha = dl_layer.alpha))
    result_cases.append(mis(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values, alpha = dl_layer.alpha))
    result_deaths.append(coverage(pi['total_deaths_lower'].values, pi['total_deaths_upper'].values, pi['total_deaths_true'].values))
    result_cases.append(coverage(pi['total_cases_lower'].values, pi['total_cases_upper'].values, pi['total_cases_true'].values))

    print(result_deaths)
    print(result_cases)

    results_deaths = results_deaths.append(pd.DataFrame(np.array(result_deaths).reshape(1,-1), columns = list(results_deaths)), ignore_index=True)
    results_cases = results_cases.append(pd.DataFrame(np.array(result_cases).reshape(1,-1), columns = list(results_cases)), ignore_index=True)
#         results_cases.append(result_cases)
    print(results_deaths.tail())
    print(results_cases.tail())
#     tf.keras.backend.clear_session()
    
print('[INFO] ---------------------- DONE -----------------------------')


results_deaths = results_deaths.iloc[1:, :]
results_cases = results_cases.iloc[1:, :]
results_deaths.to_pickle(args.country + '/' + str(args.alpha) + '/results/' + 'multiple_runs_deaths.pkl')
results_cases.to_pickle(args.country + '/' + str(args.alpha) + '/results/' + 'multiple_runs_cases.pkl')

    