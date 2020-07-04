import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from lstm_vae.vae import *

tf.compat.v1.disable_v2_behavior()
from keras.models import load_model

def encode(x, enc):
    encoded_input = enc.predict(x)
    return encoded_input

def decode(z, dec):
    mu_hat, log_sigma_hat = dec.predict(z)
    return mu_hat, log_sigma_hat

def reconstructed_probability(x, enc, dec, last_only=True):
    n_samples = 100
    mu_hat = np.zeros(x.shape)
    log_sigma_hat = np.zeros(x.shape)
    for i in range(n_samples):
        z = encode(x, enc)
        mu, log_sigma = decode(z, dec)
        mu_hat += mu
        log_sigma_hat += log_sigma
        
    mu_hat = mu_hat/n_samples
    log_sigma_hat = log_sigma_hat/n_samples
    sigma_hat = np.exp(log_sigma_hat)
    if(last_only==True):
        r_prob = norm.logpdf(x[:,-1,:], mu_hat[:,-1,:], sigma_hat[:,-1,:])
    else:
        r_prob = norm.logpdf(x, mu_hat, sigma_hat)
    
    return r_prob

def save_models(path, name, model, dataset, window, file_att):
    model.save(path+"/"+name+"_"+dataset+"_w_"+str(window)+"_"+file_att)
    
def load_models(path, name, model, dataset, window, file_att):
    model = load_model(path+"/"+name+"_"+dataset+"_w_"+str(window)+"_"+file_att)

def train_model(x_train, y_train, model, epochs, dataset, save_model=True, load_model=False, file_att=None): 
    window = x_train.shape[1]
    if load_model==True:
        print(20*"="+" Loading Model "+20*"=")
        load_models(path,'model',model,dataset,window)
    
    print(20*"="+" Training Model "+20*"=")
    from keras.callbacks import History, EarlyStopping, Callback, ReduceLROnPlateau, ModelCheckpoint

    checkpoint_path = "/checkpoint/"+dataset+"/w_"+str(window)+"_"+file_att+"/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cbs = [History(), EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001, verbose=1), 
							ModelCheckpoint(filepath=checkpoint_path,
                                                                        monitor = 'val_loss',
                                                 			save_weights_only=True,
                                                 			verbose=1, mode='min')]
    
    train_start = time.time()
    history = model.fit(x_train,[x_train,y_train],epochs = epochs, batch_size=50, validation_split=0.2, 
                                                                                    verbose = 0, 
                                                                                    callbacks=cbs)
    train_time = time.time() - train_start

    
    print(20*"="+" Training is done "+20*"=")
    
    #save scores, predictions and evaluation metrics in an external file
    #with open("result/train_history_"+dataset+"_w_"+str(window)+".pkl", "wb") as file:
    #    dump(history,file)
        
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    #hist_json_file = 'history.json' 
    #with open(hist_json_file, mode='w') as f:
    #    hist_df.to_json(f)
    
    train_hist_dir = "/result/train_history/"+dataset+"/"
    
    if os.path.exists(train_hist_dir) == False:
        os.mkdir(train_hist_dir)
    
    if os.path.exists("result") == False:
        os.mkdir("result")
    
    # or save to csv: 
    hist_csv_file = train_hist_dir+"train_history_"+dataset+"_w_"+str(window)+file_att+".csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        f.close()

    # or save to csv: 
    train_time_file = train_hist_dir+"train_time_"+dataset+"_w_"+str(window)+file_att+".txt"
    with open(train_time_file, mode='w') as f:
        f.write("Training time : "+str(train_time))
        f.close()
    
def get_score(x_train, y_train, x_test, y_test, model, enc, dec, dataset, window, file_att):
    test_start = time.time()
    print(20*"="+" Computing reconstruction probability of decoded input, mae and mse of predicted input "+20*"=")
    r_prob_train = reconstructed_probability(x_train, enc, dec)
    r_prob_test = reconstructed_probability(x_test, enc, dec)

    #run the model to get the recontructed input and predicted x_t
    p = model.predict(x_train)
    p_test = model.predict(x_test)

    #compute mse reconstruction
    p_recon_mse = np.square(x_train-p[0]).mean(-1).mean(-1)
    p_test_recon_mse = np.square(x_test-p_test[0]).mean(-1).mean(-1)

    #compute mae prediction
    p_mae = (np.abs(y_train - p[1])).mean(-1)
    p_test_mae = (np.abs(y_test - p_test[1])).mean(-1)
    
    #compute mse prediction
    p_mse = np.square(y_train-p[1]).mean(-1)
    p_test_mse = np.square(y_test-p_test[1]).mean(-1)
    
    #aggregating the reconstruction probability over x's features
    x_score = np.sum((r_prob_train),-1)
    x_test_score = np.sum((r_prob_test),-1)
    
    test_time = time.time() - test_start

    test_hist_dir = "/result/train_history/"+dataset+"/"
    
    if os.path.exists(test_hist_dir) == False:
        os.mkdir(test_hist_dir)
    
    # or save to csv: 
    test_time_file = test_hist_dir+"test_time_"+dataset+"_w_"+str(window)+file_att+".txt"
    with open(test_time_file, mode='w') as f:
        f.write("Testing time : "+str(test_time))
        f.close()

    
    return x_score, x_test_score, p_mae, p_test_mae, p_mse, p_test_mse

def get_eval_param(x_score, x_test_score, x_test_labels, mae_train, mae_test, window, dataset, beta, file_att=None,save_res=False,rec_only=False):
    print(20*"="+" Preparing train score, test score and test labels "+20*"=")
    if rec_only == True:
        train_score = x_score
        test_score = x_test_score
    elif rec_only == False:
        if str(dataset).startswith('machine'):
            beta=0.1
        else:
            beta=1.0
        train_score = x_score + beta*np.log(mae_train + 1e-6)
        test_score = x_test_score + beta*np.log(mae_test + 1e-6)

    else:
        train_score = np.log(mae_train + 1e-6)
        test_score = np.log(mae_test + 1e-6)
        
    test_labels = x_test_labels[:,-1]
    
    score_dir = "/result/scores/"+dataset+"/"
    
    if os.path.exists(score_dir) == False:
        os.mkdir(score_dir)
    
    if save_res == True:
        #save scores, predictions and evaluation metrics in an external file
        with open(score_dir+"train_scores_"+dataset+"_w_"+str(window)+"_loss_weights_"+file_att+".pkl", "wb") as file:
            dump(train_score,file)
        with open(score_dir+"test_scores_"+dataset+"_w_"+str(window)+"_loss_weights_"+file_att+".pkl", "wb") as file:
            dump(test_score,file)
        with open(score_dir+"test_labels_"+dataset+"_w_"+str(window)+"_loss_weights_"+file_att+".pkl", "wb") as file:
            dump(test_labels,file)
    
    return train_score, test_score, test_labels

def test_model(train_score, test_score, test_labels, window, dataset, level, file_att=None, save_res=False):
    print(20*"="+" Evaluating the model "+20*"=")
    #evaluating the results
    result = pot_eval(train_score, test_score, test_labels.reshape(-1), q=1e-4, level=level)

    eval_dir = "/result/eval/"+dataset+"/"
    
    if os.path.exists(eval_dir) == False:
        os.mkdir(eval_dir)
    
    if save_res == True:
        #save scores, predictions and evaluation metrics in an external file
        with open(eval_dir+"result_"+dataset+"_w_"+str(window)+"_loss_weights_"+file_att+"_"+str(level)+".pkl", "wb") as file:
            dump(result,file)
        
        _,_,a = result
        with open(eval_dir+"eval_metrics_"+dataset+"_w_"+str(window)+"_loss_weights_"+file_att+"_"+str(level)+".txt","w") as f:
            f.write(str(a))
            f.close()
    
    return result
    
