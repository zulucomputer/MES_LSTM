# MES_LSTM
A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling

This repository contains the code for the models (including the becnhmarks) presented in the paper by T. Mathonsi and T.L. van Zyl (currently under review). The code was tested on Ubuntu Linux and MacOS.

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


