# MES_RNN
A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Multivariate Time Series Forecasting

This repository contains the code for the model presented in the paper by T. Mathonsi and T.L. van Zyl. The repository includes implementation of their novel model MES_RNN, as well as the benchmark models presented in their article. It was tested on Ubuntu Linux and MacOS.

### Requirements
```
- python=3.8
- tensorflow=2.5.0
- tensorflow_probability=0.13.0
- matplotib=
- scikit-learn=
- pandas=
- satsmodels=
```


### File Structure
```
- MES_RNN/
  - model.py
- utils/
  - metrics.py
- results/
  - mes_rnn/
  - pure_rnn/
  - mlr/
  - sarimax/
  - varmax/
- run_MES_RNN.ipynb
```


