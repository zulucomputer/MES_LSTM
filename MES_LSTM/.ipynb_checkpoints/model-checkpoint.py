import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from os import mkdir, makedirs
from os.path import isdir
from glob import glob
from os import chdir
from os.path import splitext, split
import tensorflow as tf
import tensorflow_probability as tfp
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX, VARMAX, ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS



class preprocess():
    """
    preprocessing layer
    
    """
    def __init__(self,
                 data_path = 'data/owid-covid-data.csv',
                 loc = 'United Kingdom',
                 drop = ['smooth', 'new', 'per', 'tests_units', 'Unnamed: 0'],
                 collapse = ['iso_code', 'continent', 'location'],
                 thresh = 0.6,
                 feature_range = (1, 2),
                 first_time = 0):
        self.data_path = data_path
        self.loc = loc
        self.drop = drop
        self.thresh = thresh
        self.first_time = first_time
        self.collapse = collapse
        self.feature_range = feature_range
    
    def load_data(self):
        """
        loads dataset
        first_time - boolean: 0 if you have downloaded the data before
        
        """
        if self.first_time:
            makedirs(split(self.data_path)[0] + '/', exist_ok = True)
            url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv' # get latest data
            df = pd.read_csv(url)
            print(df.to_string())
            df.to_csv(self.data_path)
        else:        
            df = pd.read_csv(self.data_path)
#         print('[INFO] countries')
#         print(df.location.unique())
        df = df[df.location.str.contains(self.loc)] # select a country to model
        return(df)
    
    def clean_data(self, df):
        """
        cleans dataset
        drop - list of columns to delete
        collapse - columns to use as unique identifier
        """
        for regex in self.drop:
            df = df[df.columns.drop(list(df.filter(regex = regex)))]
        print('[INFO] data cleaned')
            
        for regex in self.collapse:
            df = df[df.columns.drop(list(df.filter(regex = regex)))]
        df.set_index(keys = 'date', drop = True, append = False, inplace = True)
        df.index.name = None        
        return df


    def fill_missing(self, df):
        print(self.missing_values(df))
        print('[INFO] filling NA')
        df = df.dropna(thresh = df.shape[0] * self.thresh, how = 'all', axis = 1)
        print(self.missing_values(df))
        
        print('[INFO] imputing categorical missing values')
        num_cols = df._get_numeric_data().columns
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(df[num_cols])
        IterativeImputer(random_state=0)
        df[num_cols] = imp.transform(df[num_cols])
        self.missing_values(df)
        return(df)
        
    def missing_values(self, df):
        """
        tracks missing values

        """
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("[INFO] the dataframe has " + str(df.shape[1]) + " columns in total and " +
               str(mis_val_table_ren_columns.shape[0]) +
                " columns that have missing values.")
        return mis_val_table_ren_columns
    
    def scale(self, df):
        """
        scale df
        ensure entire df is strictly positive 
        
        """
        df_scaler = MinMaxScaler(feature_range = self.feature_range)
        df_scaler.fit(df)
        scaled = df_scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns = df.columns, index = df.index)
        print('[INFO] data scaled')
        return scaled_df, df_scaler

        
class ES(preprocess):
    """
    exponential smoothing layer
    
    """
    
    def __init__(self,
                 internals_path = 'internals',
                 params_path = 'params',
                 alpha = 0.1,
                 loc = 'United Kingdom'):
        
        self.internals_path = internals_path
        self.params_path = params_path
        self.alpha = alpha
        self.loc = loc
    
    def get_params(self, x_i, dates):
        fit1 = ExponentialSmoothing(x_i).fit()
        fit2 = ExponentialSmoothing(x_i, trend = 'add', dates = dates).fit()
        fit3 = ExponentialSmoothing(x_i, trend = 'mul', dates = dates).fit()
        fit4 = ExponentialSmoothing(x_i, seasonal = 'add', dates = dates).fit()
        fit5 = ExponentialSmoothing(x_i, seasonal = 'mul', dates = dates).fit()
        fit6 = ExponentialSmoothing(x_i, trend = 'add', seasonal = 'add', dates = dates).fit()
        fit7 = ExponentialSmoothing(x_i, trend = 'mul', seasonal = 'add', dates = dates).fit()
        fit8 = ExponentialSmoothing(x_i, trend = 'add', seasonal = 'mul', dates = dates).fit()
        fit9 = ExponentialSmoothing(x_i, trend = 'mul', seasonal = 'mul', dates = dates).fit()

        params = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
        results=pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\gamma$",r"$l_0$","$b_0$","SSE"],
                             columns=['SES', "Add_Trend", "Mult_Trend", "Add_Seasonality", "Mult_Seasonality",
                                      'Add_Trend_Seas', 'Mult_Trend_Add_Seas', 'Add_Trend_Mult_Seas', 'Mult_Trend_Seas'])
        
        results["SES"] = [fit1.params[p] for p in params] + [fit1.sse]
        results["Add_Trend"] = [fit2.params[p] for p in params] + [fit2.sse]
        results["Mult_Trend"] = [fit3.params[p] for p in params] + [fit3.sse]
        results["Add_Seas"] = [fit4.params[p] for p in params] + [fit4.sse]
        results["Mult_Seas"] = [fit5.params[p] for p in params] + [fit5.sse]

        results["Add_Trend_Seas"] = [fit6.params[p] for p in params] + [fit6.sse]
        results["Mult_Trend_Add_Seas"] = [fit7.params[p] for p in params] + [fit7.sse]
        results["Add_Trend_Mult_Seas"] = [fit8.params[p] for p in params] + [fit8.sse]
        results["Mult_Trend_Seas"] = [fit8.params[p] for p in params] + [fit8.sse]
#         print('[INFO] exponential smoothing parameters extracted')

#         print(results)
        return results

    def get_internals(self, x_i, dates):
        fit = ExponentialSmoothing(x_i, seasonal = 'add', trend = 'add', dates = dates).fit()
        internals = pd.DataFrame(np.c_[x_i, fit.level, fit.slope, fit.season, fit.fittedvalues],
                      columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'], index = dates)
        return internals
    
    def es(self, df):
        
        internals_dict = dict()
        params_dict = dict()
        
        
        if isdir(str(self.alpha) + '/' + self.params_path) == False:
            makedirs(self.loc + '/' + str(self.alpha) + '/' + self.params_path, exist_ok = True)
            # get and save params
            for col in df.columns.to_list():
                index = df.columns.to_list().index(col)
                ind = str(index).zfill(2)
                vars()["param_df_{}".format(ind)] = pd.DataFrame(self.get_params(df.iloc[:, index], df.index))
                vars()["param_df_{}".format(ind)].to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.params_path + "/param_df_{}".format(ind) + ".pkl")
                params_dict["param_df_{}".format(ind)] = vars()["param_df_{}".format(ind)]
        else:
            # load params
            chdir(self.loc + '/' + str(self.alpha) + '/' + self.params_path)
            for file in glob("*df*"):
#                 print("...loading DataFrame - {}".format(splitext(file)[0]))
                vars()[splitext(file)[0]] = pd.read_pickle(file)
                params_dict[splitext(file)[0]] = vars()[splitext(file)[0]]
            print('[INFO] parameter data loaded')

            chdir("..")
            chdir("..")
            chdir('..')

        if isdir(self.loc + '/' + str(self.alpha) + '/' + self.internals_path) == False:
            makedirs(self.loc + '/' + str(self.alpha) + '/' + self.internals_path)
            # get and save internals
            for col in df.columns.to_list():
                i = df.columns.to_list().index(col)
                ind = str(i).zfill(2)
#                 print("...executing for column {} - {}".format(i, col))
                vars()["int_df_{}".format(ind)] = self.get_internals(df.iloc[:, i], df.index)
                vars()["int_df_{}".format(ind)].to_pickle(self.loc + '/' + self.internals_path + "/int_df_{}".format(ind) + ".pkl")
                internals_dict["int_df_{}".format(ind)] = vars()["int_df_{}".format(ind)]
            print('[INFO] internals executed')
        else:
            # load internals
            chdir(self.loc + '/' + str(self.alpha) + '/' + self.internals_path + "/")
            for file in glob("*df*"):
#                 print("...loading DataFrame - {}".format(splitext(file)[0]))
                vars()[splitext(file)[0]] = pd.read_pickle(file)
                internals_dict[splitext(file)[0]] = vars()[splitext(file)[0]]
            print('[INFO] internals loaded')

            chdir("..")
            chdir("..")
            chdir('..')
            
        return params_dict, internals_dict
    
    def deTS(sel, scaled_df, internals_dict):
        """
        detrend, deseasonalize
        
        """
        #internals for scaled_df

        df_trend = pd.DataFrame(index = scaled_df.index)
        df_seas = pd.DataFrame(index = scaled_df.index)

        for i in sorted(internals_dict.keys()):
            df_trend[i] = internals_dict[i]['$l_t$']
            df_seas[i] = internals_dict[i]['$s_t$']
            
        # smooth for scaled_df

        es_scaled = scaled_df.values - df_trend.values - df_seas.values # exponentially smoothed (detrended, deseasononalized)
        es_scaled_df = pd.DataFrame(es_scaled, columns = scaled_df.columns, index = scaled_df.index)
        
        return es_scaled_df, df_trend, df_seas # exponentially smoothed and scaled
            
        
class lstm():
    """
    deep learning layer
    this layer is also bundled with the postprocessing methods
    
    """
    
    def __init__(self,
                 train_test = 0.1,
                 valid_test = 0.75,
                 y_col = ['total_deaths', 'total_cases'], # define y variable, i.e., what we want to predict
                 n_input_train = 14, # how many samples/rows/timesteps to look in the past in order to forecast the next sample
                 b_size_train = 32, # Number of timeseries samples in each batch
                 n_input_valid = 7, # how many samples/rows/timesteps to look in the past in order to forecast the next s
                 b_size_valid = 128, # Number of timeseries samples in each batch
                 lstm_size = 150,
                 activation = 'relu',
                 optimizer = 'adam',
                 loss = 'mse',
                 epochs = 15,
                 runs = 100,
                 alpha = 0.1,
                 loc = 'United Kingdom',
                 results_path = 'results/mes_lstm/'
                ):

        
        self.train_test = train_test
        self.valid_test = valid_test
        self.y_col = y_col
        self.n_input_train = n_input_train
        self.b_size_train = b_size_train
        self.n_input_valid = n_input_valid
        self.b_size_valid = b_size_valid
        self.lstm_size = lstm_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.runs = runs
        self.alpha = alpha
        self.results_path = results_path
        self.loc = loc
        
        
        
    def split(self, es_scaled_df):
        # split train, test, valid for es_scaled_df
    
        test_size = int(len(es_scaled_df) * self.train_test) # the train data will be 90% (0.9) of the entire data
        train = es_scaled_df.iloc[:-test_size,:].copy() # copy() prevents getting: SettingWithCopyWarning
        test = es_scaled_df.iloc[-test_size:,:].copy()
        valid, test = train_test_split(test, test_size = self.valid_test, shuffle = False) # valid is 25% of 10%
        
        # split x and y only for the train data (for now)
        x_train = train.drop(self.y_col, axis = 1).copy()
        y_train = train[self.y_col].copy()
        # split x and y only for the validation data (for now)
        x_valid = valid.drop(self.y_col, axis = 1).copy()
        y_valid = valid[self.y_col].copy()
        x_test = test.drop(self.y_col, axis = 1).copy() # split x for test
        
        print('[INFO] data shape: train = {}, valid = {}, test = {}, x_train = {}, y_train = {}, x_valid = {}, y_valid = {}, x_test = {}'.format(
            train.shape, valid.shape, test.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape))
        return train, valid, test, x_train, y_train, x_valid, y_valid, x_test
    
    
    def forecast_model(self, test, x_train, y_train, x_valid, y_valid, x_test):
        """
        returns model forecast
        
        """
        n_features_valid = x_valid.shape[1]
        n_features_train = x_train.shape[1] # how many predictors/x's/features we have to predict y
        
        train_generator = TimeseriesGenerator(x_train.values, y_train.values, length = self.n_input_train, batch_size = self.b_size_train)
        valid_generator = TimeseriesGenerator(x_valid.values, y_valid.values, length = self.n_input_valid, batch_size = self.b_size_valid)

        model = Sequential()
        model.add(LSTM(self.lstm_size, activation = self.activation, input_shape = (self.n_input_train, n_features_train)))
        model.add(Dense(len(self.y_col)))
        model.compile(optimizer = self.optimizer, loss = self.loss)
        print(model.summary())
        
        model.fit_generator(train_generator, validation_data = valid_generator, epochs = self.epochs)
        train_loss_per_epoch = model.history.history['loss']
        valid_loss_per_epoch = model.history.history['val_loss']
        
        fig, ax = plt.subplots()
        ax.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label = 'train')
        ax.plot(range(len(train_loss_per_epoch)), valid_loss_per_epoch, label = 'valid')
        leg = ax.legend()
        plt.show()
        
        test_generator = TimeseriesGenerator(x_test, np.zeros(test[self.y_col].shape), length = self.n_input_train, batch_size = self.b_size_train)
        y_pred_es_scaled = model.predict(test_generator)
        return y_pred_es_scaled
    
    
    def pi_model(self, test, x_train, y_train, x_valid, y_valid, x_test):
        """
        returns model prediction intervals
        
        """
        n_features_valid = x_valid.shape[1]
        n_features_train = x_train.shape[1] # how many predictors/x's/features we have to predict y
        
        train_generator = TimeseriesGenerator(x_train.values, y_train.values, length = self.n_input_train, batch_size = self.b_size_train)
        valid_generator = TimeseriesGenerator(x_valid.values, y_valid.values, length = self.n_input_valid, batch_size = self.b_size_valid)

        model = Sequential()
        model.add(LSTM(self.lstm_size, activation = self.activation, input_shape = (self.n_input_train, n_features_train)))
        model.add(tfp.layers.DenseFlipout(len(self.y_col)))
        model.compile(optimizer = self.optimizer, loss = self.loss)
        print(model.summary())
        
        model.fit_generator(train_generator, validation_data = valid_generator, epochs = self.epochs)
        train_loss_per_epoch = model.history.history['loss']
        valid_loss_per_epoch = model.history.history['val_loss']
        
        fig, ax = plt.subplots()
        ax.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label = 'train')
        ax.plot(range(len(train_loss_per_epoch)), valid_loss_per_epoch, label = 'valid')
        leg = ax.legend()
        plt.show()
        
        test_generator = TimeseriesGenerator(x_test, np.zeros(test[self.y_col].shape), length = self.n_input_train, batch_size = self.b_size_train)

        vars()["pi_{}".format(self.y_col[0])] = pd.DataFrame()
        vars()["pi_{}".format(self.y_col[1])] = pd.DataFrame()
        
        for run in range(self.runs):
            y_pred_scaled = model.predict(test_generator)
            vars()["pi_{}".format(self.y_col[0])][str(run)] = y_pred_scaled[:, 0]
            vars()["pi_{}".format(self.y_col[1])][str(run)] = y_pred_scaled[:, 1]
            
        pi_pred_es_scaled = pd.DataFrame()
        lower_q = self.alpha/2
        upper_q = 1 - self.alpha + self.alpha/2
        pi_pred_es_scaled[self.y_col[0] + '_lower'] = np.quantile(vars()["pi_{}".format(self.y_col[0])], lower_q, axis = 1)
        pi_pred_es_scaled[self.y_col[1] + '_lower'] = np.quantile(vars()["pi_{}".format(self.y_col[1])], lower_q, axis = 1)
        pi_pred_es_scaled[self.y_col[0] + '_upper'] = np.quantile(vars()["pi_{}".format(self.y_col[0])], upper_q, axis = 1)
        pi_pred_es_scaled[self.y_col[1] + '_upper'] = np.quantile(vars()["pi_{}".format(self.y_col[1])], upper_q, axis = 1)
        
#         print(pi_pred_es_scaled.head())
        print('[INFO] prediction intervals computed')
        return pi_pred_es_scaled
    
    
    
    
    def reTS(self, y_pred_es_scaled, es_scaled, train, valid, df_trend, df_seas, df_scaler, df):
        """
        re-trend, re-seasonalize, descale
        returns forecasts and truth
        
        """
        # descale & desmooth forecast

        placehold_df = es_scaled.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = y_pred_es_scaled[:, 0]
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = y_pred_es_scaled[:, 1]
        placehold_df = placehold_df.values + df_trend.values + df_seas.values # desmooth
        
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = es_scaled.columns) # descale


        y_pred = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:]

        forecasts = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_pred' : y_pred.iloc[:, 0],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[1] +'_pred' : y_pred.iloc[:, 1]})
#         print(forecasts)
        forecasts.plot(y = [self.y_col[0] + '_pred', self.y_col[0] + '_true'])
        forecasts.plot(y = [self.y_col[1] + '_pred', self.y_col[1] + '_true'])
        
#         if isdir(self.results_path) == False:
        makedirs(self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok = True)
        forecasts.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'forecast.pkl')
        print('[INFO] forecasts saved in results folder')
        return forecasts


    def reTS_pi(self, pi_pred_es_scaled, es_scaled, train, valid, df_trend, df_seas, df_scaler, df):
        """
        re-trend, re-seasonalize, descale
        returns PIs and truth
        
        """
        # descale & desmooth lower
        placehold_df = df.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_es_scaled[self.y_col[0] + '_lower']
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_es_scaled[self.y_col[1] + '_lower']
        placehold_df = placehold_df.values + df_trend.values + df_seas.values # desmooth
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = es_scaled.columns) # descale
        pi_pred_0 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_0
        
        # descale & desmooth second predictant
        placehold_df = df.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_es_scaled[self.y_col[0] + '_upper']
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_es_scaled[self.y_col[1] + '_upper']
        placehold_df = placehold_df.values + df_trend.values + df_seas.values # desmooth
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = es_scaled.columns) # descale
        pi_pred_1 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_1
        pi = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_lower' : pi_pred_0.iloc[:, 0],
                                self.y_col[1] + '_lower' : pi_pred_0.iloc[:, 1],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_upper' : pi_pred_1.iloc[:, 0],
                                self.y_col[1] + '_upper' : pi_pred_1.iloc[:, 1],})
#         print(pi)
        pi.plot(y = [self.y_col[0] + '_true', self.y_col[0] + '_lower', self.y_col[0] + '_upper'])
        pi.plot(y = [self.y_col[1] + '_true', self.y_col[1] + '_lower', self.y_col[1] + '_upper'])
#         if isdir(self.results_path) == False:
        makedirs(self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok = True)
        pi.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'pi.pkl')
        print('[INFO] prediction intervals saved in results folder')
        return pi



    def descale(self, y_pred_scaled, scaled_df, train, valid, df_scaler, df):
        """
        descale
        returns forecasts and truth
        
        """
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = y_pred_scaled[:, 0]
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = y_pred_scaled[:, 1]
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = scaled_df.columns) # descale


        y_pred = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:]

        forecasts = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_pred' : y_pred.iloc[:, 0],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[1] +'_pred' : y_pred.iloc[:, 1]})
#         print(forecasts)
        forecasts.plot(y = [self.y_col[0] + '_pred', self.y_col[0] + '_true'])
        forecasts.plot(y = [self.y_col[1] + '_pred', self.y_col[1] + '_true'])
        
        if isdir(self.loc + '/' + str(self.alpha) + '/' + self.results_path) == False:
            mkdir(self.loc + '/' + str(self.alpha) + '/' + self.results_path)
        forecasts.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'forecast.pkl')
        print('[INFO] forecasts saved in results folder')
        return forecasts


    def descale_pi(self, pi_pred_scaled, scaled_df, train, valid, df_scaler, df):
        """
        descale
        returns PIs and truth
        
        """
        # first
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_scaled[self.y_col[0] + '_lower']
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_scaled[self.y_col[1] + '_lower']
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = scaled_df.columns) # descale
        pi_pred_0 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_0
        
        # second
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_scaled[self.y_col[0] + '_upper']
        placehold_df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:] = pi_pred_scaled[self.y_col[1] + '_upper']
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = scaled_df.columns) # descale


        pi_pred_1 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_1

        pi = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_lower' : pi_pred_0.iloc[:, 0],
                                self.y_col[1] + '_lower' : pi_pred_0.iloc[:, 1],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_upper' : pi_pred_1.iloc[:, 0],
                                self.y_col[1] + '_upper' : pi_pred_1.iloc[:, 1],})
#         print(pi)
        pi.plot(y = [self.y_col[0] + '_true', self.y_col[0] + '_lower', self.y_col[0] + '_upper'])
        pi.plot(y = [self.y_col[1] + '_true', self.y_col[1] + '_lower', self.y_col[1] + '_upper'])
        if isdir(self.loc + '/' + str(self.alpha) + '/' + self.results_path) == False:
            mkdir(self.loc + '/' + str(self.alpha) + '/' + self.results_path)
        pi.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'pi.pkl')
        print('[INFO] prediction intervals saved in results folder')
        return pi


class stats():
    """
    statistical benchmarks
    
    """
    
    def __init__(self,
                 train_test = 0.1,
                 y_col = ['total_deaths', 'total_cases'],
                 n_input_train = 14,
                 alpha = 0.1,
                 loc = 'United Kingdom',
                 results_path = 'results/varmax/'
                ):

        
        self.train_test = train_test
        self.y_col = y_col
        self.n_input_train = n_input_train
        self.alpha = alpha
        self.results_path = results_path
        self.loc = loc
        
        
    def split(self, es_scaled_df):
        
        """
        split train, test for stats
        
        """
        
        test_size = int(len(es_scaled_df) * self.train_test) # the train data will be 90% (0.9) of the entire data
        train = es_scaled_df.iloc[:-test_size,:].copy() # copy() prevents getting: SettingWithCopyWarning
        test = es_scaled_df.iloc[-test_size:,:].copy()

        # split x and y only for the train data (for now)
        x_train = train.drop(self.y_col, axis = 1).copy()
        y_train = train[self.y_col].copy()
        x_test = test.drop(self.y_col, axis = 1).copy() # split x for test

        print('[INFO] data shape: train = {}, test = {}, x_train = {}, y_train = {}, x_test = {}'.format(
            train.shape, test.shape, x_train.shape, y_train.shape, x_test.shape))
        return train, test, x_train, x_test
    
    
    def forecast_varmax(self, test, x_train, y_train, x_test):
        """
        returns model forecast & predition intervals
        
        """

        model = VARMAX(endog = y_train, exog = x_train, enforce_invertibility = False)
        model_fit = model.fit(disp = False)
#         print(model_fit.summary())
        print('[INFO] VARMAX fitting complete')
        
        model_forecast = model_fit.get_prediction(start = model.nobs, end = model.nobs + test.shape[0] - 1, exog = x_test)
        y_pred_scaled = model_forecast.predicted_mean # forecast
        pi_pred_scaled = model_forecast.conf_int(alpha = self.alpha)
#         print(y_pred_scaled, pi_pred_scaled)

        return y_pred_scaled, pi_pred_scaled

        

    def forecast_sarimax(self, test, x_train, y_train, x_test):
        """
        returns model forecast & predition intervals
        
        """
        
        y_pred_scaled = pd.DataFrame(columns = self.y_col)
        pi_pred_scaled = pd.DataFrame()

        for col in self.y_col:
            model = SARIMAX(endog = y_train[col], exog = x_train, enforce_invertibility = False)
            model_fit = model.fit(disp = False)
#             print(model_fit.summary())
            model_forecast = model_fit.get_prediction(start = model.nobs, end = model.nobs + test.shape[0] - 1, exog = x_test)
            y_pred_scaled[col] = model_forecast.predicted_mean # forecast
            pi = pd.DataFrame()
            pi = model_forecast.conf_int(alpha = self.alpha) # prediction intervals
            for pi_col in pi.columns:
                pi_pred_scaled[pi_col] = pi[pi_col]
                
#         print(y_pred_scaled, pi_pred_scaled)
        print('[INFO] SARIMAX fitting complete')
        return y_pred_scaled, pi_pred_scaled

    
    def forecast_mlr(self, test, x_train, y_train, x_test):
        """
        returns model forecast & predition intervals
        
        """
        
        y_pred_scaled = pd.DataFrame(columns = self.y_col)
        pi_pred_scaled = pd.DataFrame()

        for col in self.y_col:
            model = OLS(endog = y_train[col], exog = x_train, enforce_invertibility = False)
            model_fit = model.fit(disp = False)
#             print(model_fit.summary())
            model_forecast = model_fit.get_prediction(exog = x_test)
            model_forecast = model_forecast.summary_frame(alpha = self.alpha)
            y_pred_scaled[col] = model_forecast['mean'] # forecast
            pi = pd.DataFrame()
            pi = model_forecast[['obs_ci_lower', 'obs_ci_upper']] # prediction intervals
            pi.columns = ['lower ' + col, 'upper ' + col]
            for pi_col in pi.columns:
                pi_pred_scaled[pi_col] = pi[pi_col]
                
#         print(y_pred_scaled, pi_pred_scaled)
        print('[INFO] MLR fitting complete')
        return y_pred_scaled, pi_pred_scaled


    def descale(self, y_pred_scaled, scaled_df, train, valid, df_scaler, df):
        """
        descale & prune (latter for coparison with MES-RNN)
        returns forecasts and truth
        
        """
        placehold_df = df.copy()
        placehold_df[self.y_col[0]][-y_pred_scaled.shape[0]:] = y_pred_scaled[self.y_col[0]]
        placehold_df[self.y_col[1]][-y_pred_scaled.shape[0]:] = y_pred_scaled[self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = placehold_df.columns)
        y_pred = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:]

        forecasts = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_pred' : y_pred.iloc[:, 0],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[1] +'_pred' : y_pred.iloc[:, 1]})
#         print(forecasts)
        forecasts.plot(y = [self.y_col[0] + '_pred', self.y_col[0] + '_true'])
        forecasts.plot(y = [self.y_col[1] + '_pred', self.y_col[1] + '_true'])
        
#         if isdir(self.results_path) == False:
        makedirs(self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok = True)
        forecasts.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'forecast.pkl')
        print('[INFO] forecasts saved in results folder')
        return forecasts
    
    
    def descale_pi(self, pi_pred_scaled, scaled_df, train, valid, df_scaler, df):
        """
        descale & prune
        returns PIs and truth
        
        """
        # lower
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['lower ' + self.y_col[0]]
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['lower ' + self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = scaled_df.columns) # descale
        pi_pred_0 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_0
        
        # upper
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['upper ' + self.y_col[0]]
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['upper ' + self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = pd.DataFrame(df_scaler.inverse_transform(placehold_df), columns = scaled_df.columns) # descale
        pi_pred_1 = placehold_df[self.y_col][train.shape[0] + valid.shape[0] + self.n_input_train:] # pi_0

        pi = pd.DataFrame({self.y_col[0] + '_true' : df[self.y_col[0]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_lower' : pi_pred_0.iloc[:, 0],
                                self.y_col[1] + '_lower' : pi_pred_0.iloc[:, 1],
                                self.y_col[1] + '_true' : df[self.y_col[1]][train.shape[0] + valid.shape[0] + self.n_input_train:].values,
                                self.y_col[0] + '_upper' : pi_pred_1.iloc[:, 0],
                                self.y_col[1] + '_upper' : pi_pred_1.iloc[:, 1],})
#         print(pi)
        pi.plot(y = [self.y_col[0] + '_true', self.y_col[0] + '_lower', self.y_col[0] + '_upper'])
        pi.plot(y = [self.y_col[1] + '_true', self.y_col[1] + '_lower', self.y_col[1] + '_upper'])
#         if isdir(self.results_path) == False:
        makedirs(self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok = True)
        pi.to_pickle(self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'pi.pkl')
        print('[INFO] prediction intervals saved in results folder')
        return pi
