# MES_RNN
A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Multivariate Time Series Forecasting

This repository contains the code for the model presented in the paper by T. Mathonsi and T.L. van Zyl. The repository includes implementation of their novel model MES_RNN, as well as the benchmark models presented in their article [\cite]. It was tested on Ubuntu Linux and MacOS.

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


