"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import warnings
from utils import standardize_ts, hankel_matrix, resample_dataset
from tica import tICA

# tf.__version__ must be greater than 2
# print(len(tf.config.list_physical_devices('GPU')), "GPUs available.")
# Suppress some warnings that appeared in tf 2.2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

use_legacy = False ## for development 
if use_legacy:
    warnings.warn("Legacy mode is enabled, archival versions of functions will be used")

from sklearn.decomposition import PCA, SparsePCA, KernelPCA, FastICA

from networks import *
from regularizers import FNN

  
class TimeSeriesEmbedding:
    """Base class for time series embedding
    
    Properties
    ----------
    
    train_history : dict
        The training history of the model
    
    model : "lstm" | "mlp" | "tica" | "etd" | "delay"
        The type of model to use for the embedding.
        
    n_latent : int
        The embedding dimension
        
    n_features : int
        The number of channels in the time series
    
    **kwargs : dict
        Keyword arguments passed to the model
        
    """
    def __init__(
        self, 
        n_latent,
        time_window=10, 
        n_features=1, 
        random_state=None,
        **kwargs
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state
        
    
    def fit(self, X, y=None):
        raise AttributeError("Derived class does not contain method.")
           
    def transform(self, X, y=None):
        raise AttributeError("Derived class does not contain method.")

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the model with a time series X, and then embed X.

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.
            
        kwargs : keyword arguments passed to the model's fit() method

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        self.fit(X, **kwargs)
        return self.transform(X)    

class ETDEmbedding(TimeSeriesEmbedding):
    """Embed time series using PCA / ETD / Broomhead-King coordinates
    
    Properties
    ----------
    
    sparse : bool
        Whether to use SparsePCA or not
    kernel : "rbf" or a python function
        A nonlinear kernel to apply before performing PCA
    
    """
    def __init__(
        self, 
        *args,
        sparse=False,
        kernel=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if kernel:
            self.model = KernelPCA(
                n_components = self.n_latent, 
                kernel=kernel,
                random_state = self.random_state,
                copy_X=False
                )
        elif sparse:
            self.model = SparsePCA(
                n_components = self.n_latent,
                random_state = self.random_state
                )
        else:
            self.model = PCA(
                n_components = self.n_latent, 
                random_state = self.random_state
                )
                
    
    def fit(self, X, y=None, subsample=None):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        subsample : int or None
            If set to an integer, a random number of timepoints is selected
            equal to that integer

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """        
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)
        if subsample:
            self.train_indices, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_new = self.model.transform(X_test)
        return X_new

    
class ConstantLagEmbedding(TimeSeriesEmbedding):
    """
    Embed a time series using constant (fixed) lag between values
    A minimal diffeomorphism, as described in Takens 1981
    
    Properties
    ----------
    lag_time : int
        The constant lag time, default 1
    """
    def __init__(
        self, 
        *args,
        lag_time = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lag_time = 1
        
    def fit(self, X, y=None, lag_cutoff=None):
        """No fitting is performed when using this embedding method
        """
        pass
    
    def transform(self, X, y=None):
        tau = self.time_window*self.lag_time
        X_test = hankel_matrix(standardize_ts(X), 
                               q = tau,
                               p = len(X) - tau
                              )
        X_test = X_test[:, ::self.lag_time, :]
        return np.squeeze(X_test)

from sklearn.metrics import mutual_info_score
from scipy.signal import savgol_filter, argrelextrema
# utils_beta.py has several helper functions built-in
class AMIEmbedding(ConstantLagEmbedding):
    """Embed time series using averaged mutual information (Fraser & Swinney 1986)
    
    Properties
    ----------
    lag_cutoff : int
        The maximum time lag to consider
        
    Notes
    -----
    Potentially may speed up _mutual_information_lagged by precomputing 
    lag-1 Hankel matrix
    """
    def __init__(
        self, 
        *args,
        lag_cutoff = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)   
        self.lag_cutoff = lag_cutoff
        
    @staticmethod
    def _compute_mutual_information(x, y, bins=None):
        """Find the pairwise mutual information between two datasets
        bins : int, the number of histogram bins to use
        """
        if bins:
            b = np.histogram2d(x, y, bins)[0]
        else:
            b = np.histogram2d(x, y, len(x))[0]
        mutual_info = mutual_info_score(None, None, contingency=b)
        return mutual_info
    
    @classmethod
    def _mutual_information_lagged(self, data, max_time, bins=None):
        """Compute the mutual information at a series of delay times"""
        all_mi = list()
        for tau in range(1, max_time):
            unlagged = data[:-tau]
            lagged = np.roll(data, -tau)[:-tau]
            joint = np.vstack((unlagged, lagged))
            all_mi.append(self._compute_mutual_information(joint[0,:], joint[1,:], bins))
        return np.array(all_mi)
    
    @staticmethod
    def _find_minima(x, smoothing_radius=None):
        """
        - smoothing_radius : int, the expected periodicity of peaks.
        Uses Savitzky-Golay filtering, which preserves the locations of extrema
        """
        if smoothing_radius:
            if (smoothing_radius % 2) == 0:
                smoothing_radius += 1
            x = savgol_filter(x, smoothing_radius, 3)
        minima = argrelextrema(x, np.less)
        return minima

    
    def fit(self, X, y=None, verbose=False, bins=None, timescale=None):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_timepoints is the number of timepoints
            and n_features is the number of features.

        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """        
        Xs = standardize_ts(X)
        
        if not self.lag_cutoff:
            self.lag_cutoff = int(np.floor(len(Xs)/2))
        lagged_mi_vals = self._mutual_information_lagged(Xs, self.lag_cutoff, bins)
        if verbose:
            plt.plot(lagged_mi_vals)
        
        lag_times = self._find_minima(lagged_mi_vals, timescale)[0]
        
        lag_time = lag_times[0]
        
        self.lag_time = lag_time # overrides constant
        

class TICAEmbedding(TimeSeriesEmbedding):
    """Embed time series using tICA
    
    Properties
    ----------
    
    time_lag : int
        The number of time steps to lag coordinates before embedding
    """
    def __init__(
        self, 
        *args,
        time_lag=10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time_lag = time_lag
        if time_lag > 0:
            self.model = tICA(n_components = self.n_latent, lag_time = time_lag)
        elif time_lag == 0:
            self.model = FastICA(
                n_components = self.n_latent,
                random_state = self.random_state
                )
        else:
            raise ValueError("Time delay parameter must be greater than or equal to zero.")
    
    def fit(self, X, y=None, subsample=None):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_timepoints is the number of timepoints
            and n_features is the number of features.

        y : None
            Ignored variable.

        subsample : int or None
            If set to an integer, a random number of timepoints is selected
            equal to that integer

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """        
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)

        if subsample:
            self.train_indices, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
        if self.time_lag > 0:
            self.model.fit([np.reshape(X_train, (X_train.shape[0], -1))])
        else:
            self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        if self.time_lag > 0:
            X_new = self.model.transform([X_test])[0]
        else:
            X_new = self.model.transform(X_test)
        return X_new
       

    
class NeuralNetworkEmbedding(TimeSeriesEmbedding):
    """Base class autoencoder model for time series embedding
    
    Properties
    ----------
    
    n_latent : int
        The embedding dimension
        
    n_features : int
        The number of channels in the time series
    
    **kwargs : dict
        Keyword arguments passed to the model
        
    """
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # # Default latent regularizer is FNN
        # if np.isscalar(latent_regularizer):
        #     latent_regularizer = FNN(latent_regularizer)
    
    def fit(
        self, 
        X,
        y=None,
        subsample=None,
        tau=0,
        learning_rate=1e-3, 
        batch_size=100, 
        train_steps=200,
        loss='mse',
        verbose=0,
        optimizer="adam",
        early_stopping=False
    ):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        subsample : int or None
            If set to an integer, a random number of timepoints is selected
            equal to that integer
            
        tau : int
            The prediction time, or the number of timesteps to skip between 
            the input and output time series


        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        
        # X_train = hankel_matrix(Xs, self.time_window)
        # Split the hankel matrix for a prediction task
        X0 = hankel_matrix(Xs, self.time_window + tau)
        X_train = X0[:, :self.time_window ]
        Y_train = X0[:, -self.time_window:]
        
        
        if subsample:
            self.train_indices, _ = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
            X_train = X_train[self.train_indices]
            Y_train = Y_train[self.train_indices]


        optimizers = {
            "adam": tf.keras.optimizers.Adam(lr=learning_rate),
            "nadam": tf.keras.optimizers.Nadam(lr=learning_rate)
            # "radam": tfa.optimizers.RectifiedAdam(lr=learning_rate),
        }

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model.compile(
            optimizer=optimizers[optimizer], 
            loss=loss,
            #experimental_run_tf_function=False
        )    
        
        if early_stopping:
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)]
        else:
            callbacks = [None]
        
        self.train_history = self.model.fit(
            x=tf.convert_to_tensor(X_train),                         
            y=tf.convert_to_tensor(Y_train),
            epochs=train_steps,
            batch_size=batch_size,
            verbose=verbose
        )
             
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_new = self.model.encoder.predict(X_test)
        return X_new 

# class CausalEmbedding(NeuralNetworkEmbedding):
#     """
#     Calculates strides and input size automatically
#     """
#     def __init__(
#         self,
#         *args,
#         network_shape=[10, 10],
#         **kwargs
#     ):  
#         super().__init__(*args, **kwargs)
#         self.depth = len(network_shape)
#         print(self.n_latent)
#         stride_size = math.floor(math.log(self.time_window/self.n_latent, (2*self.depth)))
#         final_conv_size = math.ceil(self.time_window/(stride_size**(2*self.depth)))
#         time_window_new = final_conv_size*(stride_size**(2*self.depth))
#         #print(time_window_new, stride_size, final_conv_size)
#         if time_window_new != self.time_window:
#             self.time_window = time_window_new
#             print("Time window increased to ", str(time_window_new), ", an integer power",
#              "of stride size. If this is too large, decrease network depth or latent size")
#         print("Effective stride size is ", str(stride_size**2)) # each block does two downsampling
#         print("Final convolution size is ", str(final_conv_size))

#         self.model = CausalAutoencoder(
#             self.n_latent,
#             network_shape=network_shape,
#             conv_output_shape=final_conv_size,
#             strides = stride_size,
#             **kwargs
#         )

class MLPEmbedding(NeuralNetworkEmbedding):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        kwargs.pop("time_window")
        self.model = MLPAutoencoder(
            self.n_latent,
            self.time_window,
            **kwargs
        )

class LSTMEmbedding(NeuralNetworkEmbedding):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        kwargs.pop("time_window")
        if use_legacy:
            self.model = LSTMAutoencoderLegacy(
                self.n_latent,
                self.time_window,
                **kwargs
            ) 
        else:
            self.model = LSTMAutoencoder(
                self.n_latent,
                self.time_window,
                **kwargs
            )


class CausalEmbedding(NeuralNetworkEmbedding):
    def __init__(
        self,
        *args,
        strides=1, 
        network_shape=[10, 10],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        kwargs.pop("time_window")
        time_window_downsampled = math.ceil(self.time_window/(strides**(len(network_shape))))
        time_window_upsampled = time_window_downsampled*(strides**(len(network_shape)))
        if time_window_upsampled != self.time_window:
            print("non-integer stride output detected; using closest time window " + str(time_window_upsampled))
            self.time_window = time_window_upsampled
        
        self.model = CausalAE(
            self.n_latent,
            self.time_window,
            strides=strides,
            network_shape=network_shape,
            **kwargs
        )
