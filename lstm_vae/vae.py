import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, LSTM, RepeatVector
from tensorflow.python.keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives


def create_lstm_vae_multitask_variance(input_dim, 
    timesteps, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    epsilon_std=1.,
    loss_weight=[1.0,10.0],
    dropout = 0.3):

    """
    Creates an Multi-task LSTM Variational Autoencoder (VAE). Returns Full model of multi-task learning, VAE, Encoder, Generator. 

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.
        loss_weight: [float, float]. percentage of vae loss and prediction loss, respectively
        dropout: float. 


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim, return_sequences=True, name='LSTM_qnet')(x)
    h = Dense(intermediate_dim, activation='relu', name='Dense_qnet')(h)

    # VAE Z layer
    z_mean = Dense(latent_dim, name='Dense_z_mu')(h)
    z_log_sigma = Dense(latent_dim, activation='softplus', name='Dense_z_log_sigma')(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(timesteps, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + (z_log_sigma) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,), name='z_layer')([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True, name='LSTM_pnet')
    decoder_x = Dense(intermediate_dim, activation='relu', name='Dense_pnet')
    x_mean_dense = Dense(input_dim, name='Dense_x_mu')
    x_log_sigma_dense = Dense(input_dim, activation='softplus', name='Dense_x_log_sigma')
    

    h_decoded = decoder_h(z)
    h_decoded = decoder_x(h_decoded)
    
    def sampling_x(args):
        x_mean, x_log_sigma = args
        epsilon = K.random_normal(shape=(timesteps, input_dim),
                                  mean=0., stddev=epsilon_std)
        return x_mean + (x_log_sigma) * epsilon

    # decoded layer
    x_mean = x_mean_dense(h_decoded)
    x_log_sigma = x_log_sigma_dense(h_decoded)
    x_decoded_mean = Lambda(sampling_x, output_shape=(timesteps,input_dim,), name='reconstructed_x')([x_mean, x_log_sigma])
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(timesteps, latent_dim,))

    _h_decoded = decoder_h(decoder_input)
    _h_decoded = decoder_x(_h_decoded)
    
    _x_mean = x_mean_dense(_h_decoded)
    _x_log_sigma = x_log_sigma_dense(_h_decoded)
    
    #_x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, [_x_mean, _x_log_sigma])
    
    #remove z(t) from the latent sequence {z(t-w),...,z(t-1),z(t)} such that this latent representations 
    #used as information for predicting x(t)
    z_pred = Lambda(lambda z: z[:,:-1,:], name='z_pred_layer')(z)
    
    # decoder prediction
    decoder_h_pred = LSTM(intermediate_dim, return_sequences=True, name='LSTM_pred')
    decoder_x_pred = LSTM(intermediate_dim, name='LSTM_pred_2')
    x_pred = Dense(input_dim, activation = 'linear', name='x_pred')
    
    h_decoded_pred = decoder_h_pred(z_pred)
    h_decoded_pred = Dropout(dropout)(h_decoded_pred)
    h_decoded_pred = decoder_x_pred(h_decoded_pred)
    #h_decoded_pred = Dropout(dropout)(h_decoded_pred)
    x_predicted = x_pred(h_decoded_pred)
                              
    multi_task_vae = Model(x, [x_decoded_mean, x_predicted])
    
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    multi_task_vae.compile(optimizer='adam', loss=[vae_loss,'mse'], loss_weights=loss_weight)
    print('Building VAE Multitasking Model')
                           
    return multi_task_vae, vae, encoder, generator

