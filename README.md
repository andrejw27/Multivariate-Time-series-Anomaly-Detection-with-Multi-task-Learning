# Multivariate-Time-series-Anomaly-Detection-with-Multi-task-Learning

This is an attempt to develop anomaly detection in multivariate time-series of using multi-task learning. This work is done as a Master Thesis.

Abstract:

This thesis examines the effectiveness of using multi-task learning to develop a multivariate time-series anomaly detection model. 
There have been many studies on time-series anomaly detection. However, recent studies use either a reconstruction based model or a forecasting model. 
A reconstruction based model relies on the reconstruction probability, whereas a forecasting model uses prediction error to identify anomalies. 
Anomalies are either samples with low reconstruction probability or with high prediction error, relative to a predefined threshold. 
Either way, both models learn only from a single task. Therefore, this thesis attempts to combine existing models using multi-task learning. 
In addition to that, most recent studies use unsupervised learning due to the limited labeled datasets and it is also used in this thesis. 
In order to evaluate the model, the proposed model is tested on three datasets (i.e. two public aerospace datasets and a server machine dataset) and compared with three baselines (i.e. two reconstruction based models and one forecasting model). 
The results show that the proposed model outperforms all the baselines in terms of F1-score. 
In particular, the proposed model improves F1-score by 30.43\%. Overall, the proposed model tops all the baselines which are single-task learning models.

Baseline:

- [LSTM-NDT](https://github.com/khundman/telemanom) in KDD 2018 paper ["Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"](https://arxiv.org/abs/1802.04431)
- DAGMM in ICLR 2018 Conference ["Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection"](https://openreview.net/pdf?id=BJJLHbb0-)
- [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) in KDD 2019 paper ["Robust Anomaly Detection for Multivariate Time Series
through Stochastic Recurrent Neural Network"](https://dl.acm.org/doi/pdf/10.1145/3292500.3330672)

Datasets used in the experiments:
- SMAP & MSL are two public datasets from NASA. (https://github.com/khundman/telemanom)
- Server Machine Dataset (SMD) is a server machine dataset obtained at a large internet company by the authors of OmniAnomaly. (https://github.com/NetManAIOps/OmniAnomaly)

Limitation:
- The results of the baselines are obtained using the hyperparameter setup set in each resource but only a change in window length was made.
