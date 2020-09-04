"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf
import numpy as np
import warnings
from utils import standardize_ts, hankel_matrix, resample_dataset
import math

from layers import *

# # tf.__version__ must be greater than 2
# # print(len(tf.config.list_physical_devices('GPU')), "GPUs available.")
# # Suppress some warnings that appeared in tf 2.2
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# use_legacy = False ## for development 
# if use_legacy:
#     warnings.warn("Legacy mode is enabled, archival versions of functions will be used")

class CausalAE(tf.keras.Model):
    """
    A causal autoencoder model for time series
    """
    def __init__(
        self,
        n_latent,
        time_window,
        n_features=1,
        network_shape=[10, 10],
        latent_regularizer=None,
        kernel_size=3,
        dilation_scale=2,
        strides=1,
        rnn_opts=dict(),
        activation_func=tf.keras.layers.ELU(alpha=1.0),
        random_state=None,
        add_noise=False,
        **kwargs
    ):
        super().__init__()
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features

        time_window_downsampled = math.ceil(time_window/(strides**(len(network_shape))))
        time_window_upsampled = time_window_downsampled*(strides**(len(network_shape)))

        if time_window_upsampled != time_window:
            print("non-integer stride output detected; please closest time window" + str(time_window_upsampled))
            self.time_window = time_window_upsampled
        
        # Initialize state
        tf.random.set_seed(random_state)
        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5))
        for i, hidden_size in enumerate(network_shape):
            self.encoder.add(
                CausalConv1D(
                    hidden_size, 
                    kernel_size,
                    strides=strides, 
                    padding="causal",
                    dilation_rate=dilation_scale**i,
                    activation=None, 
                    name="conv" + str(i),
                    **rnn_opts
                )
            )
            #print(dilation_scale**i)
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(
            tf.keras.layers.Dense(
                n_latent,
                activation=None,
                activity_regularizer=latent_regularizer
                )
            )
        
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5))
        self.decoder.add(tf.keras.layers.Dense(units=time_window_downsampled*network_shape[-1], activation=None))
        self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(time_window_downsampled, network_shape[-1])))

        for i, hidden_unit in enumerate(network_shape[::-1][1:] + [n_features]):
            upsamp_factor = dilation_scale**(len(network_shape) - i - 1)
            #print(upsamp_factor)
            self.decoder.add(
                CausalConv1D(
                    hidden_unit, 
                    kernel_size, # kernel size
                    strides=strides, 
                    padding='same', #same?
                    dilation_rate=upsamp_factor,
                    transpose=True,
                    activation=None,
                    name="deconv" + str(i),
                    **rnn_opts
                )
            )

            self.decoder.add(tf.keras.layers.BatchNormalization())
            # no activation on final layer
            if i < len(network_shape):
                self.decoder.add(tf.keras.layers.Activation(activation_func))
    
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs


# class CausalAutoencoder(tf.keras.Model):
    
#     def __init__(
#             self,
#             n_latent,
#             time_window,
#             n_features=1,
#             network_shape=[10, 10],
#             latent_regularizer=None,
#             kernel_size=3,
#             dilation_rate=2,
#             activation_func=tf.nn.elu,
#             random_state=None,
#             **kwargs
#         ):
#             super().__init__()
#             self.n_latent = n_latent
#             #self.time_window = time_window
#             self.n_features = n_features

#             # Initialize state
#             tf.random.set_seed(random_state)

#             self.encoder = tf.keras.Sequential(
#                 [
#                 CausalCNN(n_latent, 
#                     network_shape, 
#                     kernel_size = kernel_size)
#                 ,tf.keras.layers.Layer(activity_regularizer=latent_regularizer)
#                 ])
#             self.decoder = tf.keras.Sequential([CausalCNN(n_features, 
#                                      network_shape, 
#                                      kernel_size = kernel_size)])
            
#     def call(self, inputs, training=False):
#         outputs = self.decoder(self.encoder(inputs))
#         return outputs


# class CausalAutoencoder(tf.keras.Model):
#     """
#     Consider taking the time window as an explicit argument, in order to avoid
#     the upsampling ambiguity
#     """
    
#     def __init__(
#             self,
#             n_latent,
#             #time_window, 
#             depth=3,
#             units_per_layer=10,
#             n_features=1,
#             network_shape=[10, 10],
#             latent_regularizer=None,
#             kernel_size=3,
#             dilation_rate=2,
#             strides=1,
#             activation_func=tf.nn.elu,
#             random_state=None,
#             conv_output_shape=None,
#             **kwargs
#         ):
#             super().__init__()
#             self.n_latent = n_latent
#             self.time_window = time_window
#             self.n_features = n_features
#             self.network_shape = network_shape
#             self.strides = strides


#             #self.depth = len(network_shape)

#             # Initialize state
#             tf.random.set_seed(random_state)

#             #self.even_shape = self.strides**(2*len(network_shape))
#             # factor of 2 comes from the two blocks

#             if not conv_output_shape:
#                 conv_output_shape = n_latent

#             self.encoder = tf.keras.Sequential([
#                 CausalCNN(n_latent, 
#                     network_shape, 
#                     kernel_size = kernel_size,
#                     strides=strides,
#                     ),
#                 tf.keras.layers.Reshape([-1]),
#                 tf.keras.layers.Dense(
#                     n_latent, 
#                     activation=None,
#                     activity_regularizer=latent_regularizer
#                     )
#                 ])

#             self.decoder = tf.keras.Sequential([
#                 tf.keras.layers.Dense(
#                     conv_output_shape*n_latent, 
#                     activation=None
#                     ),
#                 tf.keras.layers.Reshape([-1, n_latent]),
#                 CausalCNN(n_features, 
#                     network_shape, 
#                     kernel_size = kernel_size,
#                     strides=strides,
#                     transpose=True
#                     )
#                 ])
            
#     def call(self, inputs, training=False):
#         # outputs = self.decoder(
#         #     self.encoder(
#         #         tf.pad(inputs, [[0,0], [self.safe_pad, 0], [0, 0]])
#         #         )
#         #     )
#         # return outputs[:, self.safe_pad:, :]
#         outputs = self.decoder(self.encoder(inputs))
#         return outputs
        






class MLPAutoencoder(tf.keras.Model):
    """
    A fully-connected autoencoder model for time series
    """
    def __init__(
        self,
        n_latent,
        time_window,
        n_features=1,
        network_shape=[10, 10],
        latent_regularizer=None,
        rnn_opts=dict(),
        activation_func=tf.keras.layers.ELU(alpha=1.0),
        random_state=None,
        **kwargs
    ):
        super(MLPAutoencoder, self).__init__()
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        
        # Initialize state
        tf.random.set_seed(random_state)
        
        # Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(time_window, n_features)))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(time_window,))) # smooths the output
        for hidden_size in network_shape:
            self.encoder.add(tf.keras.layers.Dense(hidden_size, **rnn_opts))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
        self.encoder.add(tf.keras.layers.Dense(n_latent, input_shape=(time_window,), **rnn_opts))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(
            tf.keras.layers.Reshape(
                (n_latent,),  
                activity_regularizer=latent_regularizer
            )
        )

        ## Decoder
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Flatten())
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(n_latent,)))
        for hidden_size in network_shape[::-1]:
            self.decoder.add(tf.keras.layers.Dense(hidden_size,  **rnn_opts))
            self.decoder.add(tf.keras.layers.BatchNormalization())
            self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Dense(time_window*n_features, **rnn_opts))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        #self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Reshape((time_window, n_features)))
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs

class MLPAutoencoderLegacy(tf.keras.Model):
    """
    A fully-connected autoencoder model for time series
    """
    def __init__(
        self,
        n_latent,
        time_window,
        n_features=1,
        network_shape=[10, 10],
        latent_regularizer=None,
        rnn_opts=dict(),
        activation_func=tf.keras.layers.ELU(alpha=1.0),
        random_state=None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        
        # Initialize state
        tf.random.set_seed(random_state)
        
        # Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(time_window, n_features)))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(time_window,))) # smooths the output
        
        for  hidden_size in network_shape:
            self.encoder.add(tf.keras.layers.Dense(hidden_size, **rnn_opts))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
    
        self.encoder.add(tf.keras.layers.Dense(self.n_latent, input_shape=(self.time_window,), **rnn_opts))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        #enc.add(tf.keras.layers.Activation(activation_func))

        self.encoder.add(tf.keras.layers.Reshape((self.n_latent,), 
                                                 activity_regularizer=latent_regularizer))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Flatten())
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(n_latent,)))

        for hidden_size in network_shape[::-1]:
            self.decoder.add(tf.keras.layers.Dense(hidden_size,  **rnn_opts))
            self.decoder.add(tf.keras.layers.BatchNormalization())
            self.decoder.add(tf.keras.layers.Activation(activation_func))


        self.decoder.add(tf.keras.layers.Dense(self.time_window*self.n_features, **rnn_opts))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation(activation_func))

        self.decoder.add(tf.keras.layers.Reshape((self.time_window, self.n_features)))
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs
    
    
class LSTMAutoencoder(tf.keras.Model):
    """
    An LSTM autoencoder model for time series
    """
    def __init__(
        self,
        n_latent,
        time_window,
        n_features=1,
        network_shape=[],
        latent_regularizer=None,
        rnn_opts=dict(),
        activation_func=tf.keras.layers.ELU(alpha=1.0),
        random_state=None,
        **kwargs
    ):
        super(LSTMAutoencoder, self).__init__()
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        
        # Initialize state
        tf.random.set_seed(random_state)
        
        # Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(time_window, n_features)))
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5)) # smooths the output
        
        for i, hidden_size in enumerate(network_shape):
            self.encoder.add(
                tf.keras.layers.LSTM(
                    hidden_size,
                    #input_shape=(time_window, n_features),
                    return_sequences=True,
                    name="lstm_encoder_"+str(i),
                    **rnn_opts
                )
            )
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
        self.encoder.add(
            tf.keras.layers.LSTM(
                n_latent,
                #input_shape=(time_window, n_features),
                return_sequences=False,
                name="lstm_encoder_final",
                **rnn_opts
            )
        )
        self.encoder.add(tf.keras.layers.BatchNormalization(activity_regularizer=latent_regularizer))
            
        ## Decoder
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(n_latent,)))
        self.decoder.add(tf.keras.layers.RepeatVector(time_window))
        self.decoder.add(
            tf.keras.layers.LSTM(
                n_latent,
                #input_shape=(time_window, n_features),
                return_sequences=True,
                name="lstm_decoder_initial",
                **rnn_opts
            )
        )
        
        
        for i, hidden_size in enumerate(network_shape[::-1]):
            self.decoder.add(
                tf.keras.layers.LSTM(
                    hidden_size, 
                    return_sequences=True, 
                    go_backwards=True, 
                    name="lstm_decoder_"+str(i),
                    **rnn_opts
                )
            )
        self.decoder.add(
            tf.keras.layers.LSTM(
                n_features, 
                return_sequences=True, 
                go_backwards=True, 
                name="lstm_decoder_final"),
                **rnn_opts
        )
        self.decoder.add(tf.keras.layers.BatchNormalization())
        #self.decoder.add(tf.keras.layers.Activation(activation_func))
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs
    
class LSTMAutoencoderLegacy(tf.keras.Model):
    """
    An LSTM autoencoder model, based on the architecture used in the original 
    FNN preprint
    """
    def __init__(
        self,
        n_latent,
        time_window,
        n_features=1,
        network_shape=[],
        latent_regularizer=None,
        rnn_opts=dict(),
        activation_func=tf.keras.layers.ELU(alpha=1.0),
        random_state=None,
        **kwargs
    ):
        super().__init__()
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        
        # Initialize state
        tf.random.set_seed(random_state)
        
        # Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(time_window, n_features)))
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5))  # smooths the output
        self.encoder.add(
            tf.keras.layers.LSTM(
                self.n_latent,
                input_shape=(self.time_window, self.n_features),
                return_sequences=False,
                **rnn_opts
            )
        )
        self.encoder.add(tf.keras.layers.BatchNormalization(activity_regularizer=latent_regularizer))
        
        
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.RepeatVector(self.time_window))
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5))
        self.decoder.add(
            tf.keras.layers.LSTM(
                self.n_latent, return_sequences=True, go_backwards=True, **rnn_opts
            )
        )
        self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs
 
