# MES_LSTM
A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling

This repository contains the code for the models (including the becnhmarks) presented in the paper by T. Mathonsi and T.L. van Zyl. The code was tested on Ubuntu Linux and MacOS.

```run_MES_LSTM.ipynb``` Executes the model for a single country. ```run_simultaneous.sh``` Is the bash script which parallelizes ```run_multi_MES_LSTM.py``` and executes the model for any number of nations or region, with multiple independent trials. The latter accepts command-line arguments. ```results.ipynb``` Reproduces the figures and tables from the paper.

### Requirements
```
- python=3.9.6
- tensorflow=2.4.1
- tensorflow_probability=0.12.2
- matplotib=3.4.2
- scikit-learn=0.24.2
- pandas=1.3.2
- satsmodels=0.12.2
- numpy=1.19.2
```


### File Structure
```
- MES_LSTM/
  - model.py
- utils/
  - metrics.py
- country_0
- country_1
- ...
- country_n
  - results/
    - mes_lstm/
    - lstm/
    - mlr/
    - sarimax/
    - varmax/
- results.ipynb
- run_simultaneous.sh
- run_multi_MES_LSTM.py

```

### Suggested Citation
```
@Article{forecast4010001,
AUTHOR = {Mathonsi, Thabang and van Zyl, Terence L.},
TITLE = {A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling},
JOURNAL = {Forecasting},
VOLUME = {4},
YEAR = {2022},
NUMBER = {1},
PAGES = {1--25},
URL = {https://www.mdpi.com/2571-9394/4/1/1},
ISSN = {2571-9394},
ABSTRACT = {Hybrid methods have been shown to outperform pure statistical and pure deep learning methods at forecasting tasks and quantifying the associated uncertainty with those forecasts (prediction intervals). One example is Exponential Smoothing Recurrent Neural Network (ES-RNN), a hybrid between a statistical forecasting model and a recurrent neural network variant. ES-RNN achieves a 9.4% improvement in absolute error in the Makridakis-4 Forecasting Competition. This improvement and similar outperformance from other hybrid models have primarily been demonstrated only on univariate datasets. Difficulties with applying hybrid forecast methods to multivariate data include (i) the high computational cost involved in hyperparameter tuning for models that are not parsimonious, (ii) challenges associated with auto-correlation inherent in the data, as well as (iii) complex dependency (cross-correlation) between the covariates that may be hard to capture. This paper presents Multivariate Exponential Smoothing Long Short Term Memory (MES-LSTM), a generalized multivariate extension to ES-RNN, that overcomes these challenges. MES-LSTM utilizes a vectorized implementation. We test MES-LSTM on several aggregated coronavirus disease of 2019 (COVID-19) morbidity datasets and find our hybrid approach shows consistent, significant improvement over pure statistical and deep learning methods at forecast accuracy and prediction interval construction.},
DOI = {10.3390/forecast4010001}
}}

```


