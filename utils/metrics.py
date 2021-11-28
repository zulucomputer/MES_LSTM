from numpy import count_nonzero, nansum, isnan, diff, abs, mean, sqrt, array

# alpha = 0.05 # default

def rmse(y, y_hat):
    return sqrt(((y_hat - y) ** 2).mean())

def smape(y, y_hat):
    tmp = 2 * abs(y_hat - y) / (abs(y) + abs(y_hat))
    len_ = count_nonzero(~isnan(tmp))
    if len_ == 0 and nansum(tmp) == 0: # deals with a special case
        return 100
    return 100 / len_ * nansum(tmp)

def mase(y_train, y, y_hat):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error.
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        y_train: the series used to train the model, 1d numpy array
        y: the test series to predict, 1d numpy array or float
        y_hat: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
    
    """
    n = y_train.shape[0]
    d = abs(diff(y_train)).sum()/(n-1)
    
    errors = abs(y - y_hat)
    return errors.mean()/d

def mis(lower_PI, upper_PI, actual, alpha):
#    alpha = alpha/100 # percentile and this score use different scales for alpha
    s = list()
    for i in range(len(lower_PI)):
        if actual[i] < lower_PI[i]:
            s.append(upper_PI[i] - lower_PI[i] + (2/alpha)*(lower_PI[i] - actual[i]))
        elif lower_PI[i] <= actual[i] <= upper_PI[i]:
            s.append(upper_PI[i] - lower_PI[i])
        else:
            s.append(upper_PI[i] - lower_PI[i] + (2/alpha)*(actual[i] - upper_PI[i]))
    s = array(s)
    return mean(s)

def coverage(lower_PI, upper_PI, actual):
    s = list()
    for i in range(len(lower_PI)):
        if lower_PI[i] <= actual[i] <= upper_PI[i]:
            s.append(1)
        else:
            s.append(0)
    return 100 * mean(s) # percentage coverage

def diebold_mariano_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

def r2_all (model, X_train, y_train, X_test, y_test, y_pred):
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    r2_valid = r2_score(y_test, y_pred)
    print(f'R2 Train: {r2_train}, R2 Test: {r2_test}, R2 Valid: {r2_valid}' )
    return r2_train, r2_test, r2_valid

def mae_all (model, X_train, y_train, X_test, y_test, y_pred):
    from sklearn.metrics import mean_absolute_error
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    mae_valid = mean_absolute_error(y_test, y_pred)
    print(f'MAE Train: {mae_train}, MAE Test: {mae_test}, MAE Valid: {mae_valid}' )
    return mae_train, mae_test, mae_valid

def evs_all (model, X_train, y_train, X_test, y_test, y_pred, multioutput='uniform_average' ):
    from sklearn.metrics import explained_variance_score
    evs_train = explained_variance_score(y_train, model.predict(X_train), multioutput=multioutput)
    evs_test = explained_variance_score(y_test, model.predict(X_test), multioutput=multioutput)
    evs_valid = explained_variance_score(y_test, y_pred, multioutput=multioutput)
    print(f'EVS Train: {evs_train}, EVS Test: {evs_test}, EVS Valid: {evs_valid}' )
    return evs_train, evs_test, evs_valid

def rmse_all (model, X_train, y_train, X_test, y_test, y_pred):
    from numpy import sqrt 
    from sklearn.metrics import mean_squared_error
    rmse_train = sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_test = sqrt(mean_squared_error(y_test, model.predict(X_test)))
    rmse_valid = sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE Train: {rmse_train}, RMSE Test: {rmse_test}, RMSE Valid: {rmse_valid}' )
    return rmse_train, rmse_test, rmse_valid

def rmsle_all (model, X_train, y_train, X_test, y_test, y_pred):
    from numpy import sqrt 
    from sklearn.metrics import mean_squared_log_error
    rmsle_train = sqrt(mean_squared_log_error(y_train, model.predict(X_train)))
    rmsle_test = sqrt(mean_squared_log_error(y_test, model.predict(X_test)))
    rmsle_valid = sqrt(mean_squared_log_error(y_test, y_pred))
    print(f'RMSLE Train: {rmsle_train}, RMSLE Test: {rmsle_test}, RMSLE Valid: {rmsle_valid}' )
    return rmsle_train, rmsle_test, rmsle_valid


