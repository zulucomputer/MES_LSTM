# MES_LSTM
A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Multivariate Time Series Forecasting

This repository contains the code for the models (including the becnhmarks) presented in the paper by T. Mathonsi and T.L. van Zyl [\cite]. It was tested on Ubuntu Linux and MacOS.

```run_MES_LSTM.ipynb``` Executes the model for a single country. ```run_simultaneous.sh``` Is the bash script which parallelizes ```run_multi_MES_LSTM.py``` and executes the model for any number of nations or region, with multiple independent trails. The latter accepts command-line arguments.

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
    - mes_rnn/
    - pure_rnn/
    - mlr/
    - sarimax/
    - varmax/
- run_MES_LSTM.ipynb
- run_simultaneous.sh
- run_multi_MES_LSTM.py

```


