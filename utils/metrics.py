from numpy import count_nonzero, nansum, isnan, diff, abs, mean

alpha = 0.05 # default

def smape(y, y_hat):
    tmp = 2 * abs(y_hat - y) / (abs(y) + abs(y_hat))
    len_ = count_nonzero(~isnan(tmp))
    if len_ == 0 and nansum(tmp) == 0: # Deals with a special case
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

def mis(lower_PI, upper_PI, actual, alpha = alpha):
#    alpha = alpha/100 # percentile and this score use different scales for alpha
    s = list()
    for i in range(len(lower_PI)):
        if actual[i] < lower_PI[i]:
            s.append(upper_PI[i] - lower_PI[i] + (2/alpha)*(lower_PI[i] - actual[i]))
        elif lower_PI[i] <= actual[i] <= upper_PI[i]:
            s.append(upper_PI[i] - lower_PI[i])
        else:
            s.append(upper_PI[i] - lower_PI[i] + (2/alpha)*(actual[i] - upper_PI[i]))
    return mean(s)

def coverage(lower_PI, upper_PI, actual):
    s = list()
    for i in range(len(lower_PI)):
        if lower_PI[i] <= actual[i] <= upper_PI[i]:
            s.append(1)
        else:
            s.append(0)
    return 100 * mean(s) # percentage coverage
