# MES_LSTM
A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Anomaly Detection

This repository contains the code for the models (including the benchmarks) presented in the papers by T. Mathonsi and T.L. van Zyl. The code was tested on Ubuntu Linux and MacOS.

```run_simultaneous.sh``` Is the bash script which parallelizes ```run_multi_MES_LSTM.py``` and executes the model for the chosen application, with multiple independent trials. The latter accepts command-line arguments. ```results.ipynb``` Reproduces the figures and tables from the papers.

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

### Suggested Citations

#### Forecasting
```
@Article{forecast4010001,
AUTHOR = {Mathonsi, Thabang and van Zyl, Terence L.},
TITLE = {A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling},
JOURNAL = {Forecasting},
VOLUME = {4},
YEAR = {2022},
NUMBER = {1},
PAGES = {1--25},
DOI = {10.3390/forecast4010001}
}

```

#### Anomaly Detection
```
@article{s00521-021-06697-x,
  title={Multivariate anomaly detection based on prediction intervals constructed using deep learning},
  author={Mathonsi, Thabang and {van Zyl}, Terence L},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2022},
  publisher={Springer},
  doi = {10.1007/s00521-021-06697-x}
}
```

Note: As of September 2022, this repository is no longer maintained.

