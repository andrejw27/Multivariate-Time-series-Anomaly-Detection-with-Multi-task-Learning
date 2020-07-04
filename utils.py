import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.autograd import Variable
import pandas as pd
import math
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error as mae, roc_auc_score, auc
from keras.optimizers import SGD, Adam
import random
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#functions for writing data into files
import ast
import csv
import sys
from pickle import dump
from spot import SPOT


# process a multivariate data using sliding window
def sliding_window(data, window_length):
    X, y = list(), list()
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + window_length
        # check if we are beyond the dataset
        if end_ix > len(data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix, :], data[i+1:end_ix+1, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))

def preprocess(dataset, window_length):
    print("="*10 + "Reading and Processing Dataset" + "="*10)
    
    x_train_,x_test_,x_anom_labels = get_data(dataset)

    #transform into dataframe to analyze it and drop columns whose mean is 0 
    x_test_df = array_to_df(x_test_)
    x_train_df = array_to_df(x_train_)
    
    #get column index with mean 0
    nontrain=(x_train_df.mean()==0).to_numpy()
    nontest=(x_test_df.mean()==0).to_numpy()
    nontrain_idx = np.where(nontrain==True)[0]
    nontest_idx = np.where(nontest==True)[0]
    drop_idx = np.intersect1d(nontrain_idx, nontest_idx)
    
    x_train_df.drop(x_train_df.columns[list(drop_idx)],axis=1, inplace=True)
    x_test_df.drop(x_test_df.columns[list(drop_idx)],axis=1, inplace=True)
    
    x_train_ = x_train_df.to_numpy()
    x_test_ = x_test_df.to_numpy()
    
    #min 0 max 1
    x_train = MinMaxScaler().fit_transform(x_train_)
    x_test = MinMaxScaler().fit_transform(x_test_)
    #or mean 0 std 1
    #x_train = StandardScaler().transform(x_train_)
    #x_test = StandardScaler().transform(x_test_)
    
    x_anom_labels_df = pd.DataFrame(x_anom_labels,columns=['label'])
    di = {False: 0, True: 1}
    x_anom_labels = x_anom_labels_df['label'].map(di).to_numpy()
    x_normal_labels = np.zeros(len(x_train))
    X_normal, y_normal = sliding_window(x_train, window_length)
    X_anomaly, y_anomaly = sliding_window(x_test, window_length)
    X_normal_labels, y_normal_labels = sliding_window(x_normal_labels.reshape((-1,1)), window_length)
    X_anom_labels, y_anom_labels = sliding_window(x_anom_labels.reshape((-1,1)), window_length)
    #print('X_normal shape :',X_normal.shape)
    #print('X_anomaly shape :', X_anomaly.shape)
    #print('X_anom_labels shape :', X_anom_labels.shape)
    
    return X_normal, y_normal, X_anomaly, y_anomaly, X_normal_labels, X_anom_labels


#def preprocess_data(dataset, window_length):

def get_data(prefix, dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=False, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    #print("train: ", train_start, train_end)
    #print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    print(x_dim)
    #prefix = "/home/wijaya/Thesis/Thesis/Thesis Dataset"
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = np.asarray(pickle.load(f)).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = np.asarray(pickle.load(f)).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = np.asarray(pickle.load(f)).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    #print("train set shape: ", train_data.shape)
    #print("test set shape: ", test_data.shape)
    #print("test set label shape: ", test_label.shape)
    return train_data,test_data, test_label


# convert array to dataframe to get statistical of each feature easier (mean, std of each feature of FCN output)
def array_to_df(data):
    FCN_out_df = pd.DataFrame(data=data[0:,0:],    # values
                              index=[i for i in range(data.shape[0])],    # 1st column as index
                              columns=['f'+str(i) for i in range(data.shape[1])])
    return FCN_out_df

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=True)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = -np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return pred,label,{
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency
    }


def get_best_f1(test_score, test_labels, thresholds):
    Precision,Recall,F1_score,TP,TN,FP,FN=[],[],[],[],[],[],[]
    TPR, FPR = [],[]
    for i,threshold in enumerate(thresholds):
        pred, p_latency = adjust_predicts(test_score, test_labels.reshape(-1), threshold, calc_latency=True)
        p_t = calc_point2point(pred, test_labels.reshape(-1))

        #detected anomalies (True Positive)
        TP.append(p_t[3])
        #True Negative on test data
        TN.append(p_t[4])
        #detected anomalies on normal test data (False Positive)
        FP.append(p_t[5])
        #False Negative on anomaly data
        FN.append(p_t[6])

        Precision.append(p_t[1])
        Recall.append(p_t[2])
        F1_score.append(p_t[0])

        TPR_i = p_t[2] #TPR == Recall
        FPR_i = p_t[5]/(p_t[5]+p_t[4])
        TPR.append(TPR_i)
        FPR.append(FPR_i) 
            
    #get index for the best result based on F1 metrics
    idx = np.nanargmax(F1_score)
    #print results
    print("Resuts of Anomaly Detector")
    print("True Positive: ",TP[idx])
    print("True Negative: ",TN[idx])
    print("False Positive: ",FP[idx])
    print("False Negative: ",FN[idx])
    print("Precision: ",Precision[idx])
    print("Recall: ",Recall[idx])
    print("F1-score: ",F1_score[idx])
    return Precision,Recall,F1_score,TP,TN,FP,FN, thresholds, TPR, FPR, idx
