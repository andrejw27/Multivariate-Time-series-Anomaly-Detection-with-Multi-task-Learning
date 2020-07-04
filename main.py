import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
from func_utils import *
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)

if __name__ == "__main__":
    #data processing
    prefix = "Thesis Dataset"
    datasets = ['MSL'] # SMAP,MSL,SMD (machine-1-1,....)
    
    #use window = np.arange(10,110,10) to study the effect of different window size
    window = [100] 
    
    #alpha is the loss weight assigned to the forecasting loss, and 1-alpha is assigned to the reconstruction loss
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for d, dataset in enumerate(datasets):
        for i, window_length in enumerate(window):
            for j, alpha in enumerate(alphas): 
                tf.reset_default_graph()
                print(10*"="+"Training Dataset "+str(dataset)+10*"=")  
                print(10*"="+"Training Window "+str(window_length)+10*"=")
                print(10*"="+"Loss weight alpha "+str([(1-alpha), alpha])+10*"=")
	          
		#pre-processing the dataset
                x_train, y_train_, x_test, y_test_, x_train_labels, x_test_labels = preprocess(dataset, window_length)
		
		#take the last x, i.e. xt as the target for the forecasting model in y_train as well as in y_test 
                y_train = x_train[:,-1,:]
                y_test = x_test[:,-1,:]
		
		#create folder to store the results
                datasets = dataset+"_"+str(window_length)

                #Hyperparameters
                input_dim = x_train.shape[-1] # 
                timesteps = x_train.shape[1] # 
                batch_size = 1
                intermediate_dim = 100
                latent_dim = 3
                epochs = 30
                loss_weights = [(1-alpha), alpha]
               

                #creating model
                model, vae, enc, dec = create_lstm_vae_multitask_variance(input_dim, 
                                                             timesteps=timesteps, 
                                                             batch_size=batch_size, 
                                                             intermediate_dim=intermediate_dim,
                                                             latent_dim=latent_dim,
                                                             epsilon_std=1.,
                                                             loss_weight=loss_weights,
                                                             dropout = 0.3)
                
		#additional attribute on saved file
                file_att=str(loss_weights)

                #start training model and save model after training is done
                train_model(x_train, y_train, model, epochs, datasets, save_model=True, 
                                                                    load_model=False, 
                                                                    file_att=file_att)

                #computing reconstruction probability,reconstruction error and prediction error
                x_score, x_test_score, mae_train, mae_test, mse_train, mse_test = get_score(x_train, y_train, x_test, y_test, 
                                                                                        model, enc, dec, datasets, window_length,
                                                                                        file_att)

	    	#rec is an indicator passed to the get_eval_param so that the function knows which mode is actually on.
		#rec=True if the model only learns from reconstruction model
		#rec=None if the model only learns from the forecasting model
		#rec=False if the model learns from both models
                if alpha == 0:
                    rec = True
                elif alpha == 1:
                    rec = None
                else:
                    rec = False

                #preparing train_score, test_score and test_labels
                train_score, test_score, test_labels = get_eval_param(x_score, x_test_score, x_test_labels, mae_train, mae_test, 
                                                                                                          window_length,
                                                                                                          datasets,
                                                                                                          alpha,
                                                                                                          file_att=file_att,
                                                                                                          save_res=True,
                                                                                                          rec_only=rec)

		#lvl is the assumption of extreme quantile of the anomalies in the dataset.
                if dataset=='MSL':
                   lvl = np.arange(0.001,0.1,0.001)
                elif dataset=='SMAP':
                   lvl = np.arange(0.01,0.1,0.01)
                else:
                   lvl = np.arange(0.0005, 0.1, 0.0005)
		
                #grid search to find the best extreme quantile assumption 
                f1_max = 0
                lvl_max = 0
                for i, level in enumerate(lvl):
                    #evaluating the model
                    _,_,result = test_model(train_score, test_score, test_labels, window_length, datasets, level=level, file_att=file_att, 
                                                                                        save_res=False)
		    #store the current max f1-score along with the respective lvl 
                    if(result['pot-f1'])>f1_max:
                        f1_max = result['pot-f1']
                        lvl_max = level
		
		#get the final results using the best extreme quantile assumption
                result = test_model(train_score, test_score, test_labels, window_length,datasets, level=lvl_max, file_att=file_att, 
                                                                                        save_res=True)



