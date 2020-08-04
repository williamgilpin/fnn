"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf
import numpy as np
import warnings
from utils import standardize_ts, hankel_matrix
from tica import tICA

# tf.__version__ must be greater than 2
# print(len(tf.config.list_physical_devices('GPU')), "GPUs available.")
# Suppress some warnings that appeared in tf 2.2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

use_legacy = False ## for development 
if use_legacy:
    warnings.warn("Legacy mode is enabled, archival versions of functions will be used")

from sklearn.decomposition import PCA, SparsePCA, KernelPCA, FastICA

    
class FNN(tf.keras.regularizers.Regularizer):
    """An activity regularizer that penalizes false-nearest-neighbors
    
    Parameters
    ----------
    strength : float
        The relative strength of the regularizer
    k : int
        The number of neighbors to use for distance calculation. 
        Traditionally set equal to one
    
    """

    def __init__(self, strength, batch_size=1, k=1):
        self.strength = strength
        self.k = k

    def __call__(self, x):
        return self.strength * loss_false(x, k=self.k)

    
class DeCov(tf.keras.regularizers.Regularizer):
    """An activity regularizer that enforces orthogonality
    Cogswell et al. ICLR 2016
    
    Parameters
    ----------
    strength : float
        The relative strength of the regularizer
    """

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * loss_cov(x)
    
L1Reg = tf.keras.regularizers.L1 
# Alias of built-in for API consistency


class CausalAutoencoder(tf.keras.Model):
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
        
        self.encoder = tf.keras.Sequential()
        for i, hidden_size in enumerate(network_shape):
            self.encoder.add(
                tf.keras.layers.Conv1D(
                    hidden_size, 
                    kernel_size,
                    strides=1, 
                    padding="causal",
                    dilation_rate=dilation_scale**i,
                    activation=None, 
                    name="conv" + str(i),
                    **rnn_opts
                )
            )

            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(n_latent))
        
        
        self.decoder = tf.keras.Sequential()

        self.decoder.add(tf.keras.layers.Dense(units=time_window*network_shape[-1], activation=None))
        self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(time_window, network_shape[-1])))

        for i, hidden_unit in enumerate(network_shape[::-1] + [n_features]):
            upsamp_factor = dilation_scale**(len(network_shape) - i)

            self.decoder.add(
                tf.keras.layers.Conv1DTranspose(
                    hidden_unit, 
                    kernel_size, # kernel size
                    strides=1, 
                    padding='same', #same?
                    dilation_rate=upsamp_factor,
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
        
        #self.decoder.add(tf.keras.layers.RepeatVector(time_window))
        
#         self.decoder.add(tf.keras.layers.Dense(n_latent*time_window))
#         self.decoder.add(tf.keras.layers.Reshape((n_latent, time_window)))
        
#         self.decoder.add(tf.keras.layers.RepeatVector(time_window))
#         self.decoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_latent)))
        
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
        **kwargs
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
    
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
            self.model = KernelPCA(n_components = self.n_latent, kernel=kernel)
        elif sparse:
            self.model = SparsePCA(n_components = self.n_latent)
        else:
            self.model = PCA(n_components = self.n_latent)
                
    
    def fit(self, X, y=None):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """        
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, len(Xs) - self.time_window, q=self.time_window)
        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_new = self.model.transform(X_test)[self.time_window:]
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
                               len(X) - tau, 
                               q = tau)
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
            self.model = FastICA(n_components = self.n_latent)
        else:
            raise ValueError("Time delay parameter must be greater than or equal to zero.")
    
    def fit(self, X, y=None):
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
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, len(Xs) - self.time_window, q=self.time_window)
        if self.time_lag > 0:
            self.model.fit([np.reshape(X_train, (X_train.shape[0], -1))])
        else:
            self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        if self.time_lag > 0:
            X_new = self.model.transform([X_test])[0][self.time_window:]
        else:
            X_new = self.model.transform(X_test)[self.time_window:]
        return X_new
       

class LSTMEmbedding(TimeSeriesEmbedding):
    """An LSTM model for embedding time series
    
    Properties
    ----------
    
    n_latent : int
        The embedding dimension
        
    time_window : int
        The number of past timesteps to use to embed each
        point
        
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
        
        
        if use_legacy:
            self.model = LSTMAutoencoderLegacy(
                self.n_latent,
                **kwargs
            ) 
        else:
            self.model = LSTMAutoencoder(
                self.n_latent,
                **kwargs
            )
    
    def fit(
        self, 
        X,
        y=None,
        loss="mse",
        learning_rate=1e-3, 
        batch_size=100, 
        train_steps=200,
        verbose=0
    ):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, len(Xs) - self.time_window, q=self.time_window)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
            loss=loss,
            #experimental_run_tf_function=False
        )    
        
        self.train_history = self.model.fit(
            x=tf.convert_to_tensor(X_train),                         
            y=tf.convert_to_tensor(X_train),
            epochs=train_steps,
            batch_size=batch_size,
            verbose=verbose
        )
             
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_new = self.model.encoder.predict(X_test)[self.time_window:]
        return X_new      

class MLPEmbedding(TimeSeriesEmbedding):
    """An MLP model for embedding a time series
    
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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = MLPAutoencoder(
            self.n_latent,
            **kwargs
        )
    
    def fit(
        self, 
        X,
        y=None,
        loss="mse",
        learning_rate=1e-3, 
        batch_size=100, 
        train_steps=200,
        verbose=0
    ):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, len(Xs) - self.time_window, q=self.time_window)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
            loss=loss,
            #experimental_run_tf_function=False
        )    
        
        self.train_history = self.model.fit(
            x=tf.convert_to_tensor(X_train),                         
            y=tf.convert_to_tensor(X_train),
            epochs=train_steps,
            batch_size=batch_size,
            verbose=verbose
        )
             
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_new = self.model.encoder.predict(X_test)[self.time_window:]
        return X_new      
    
    
class CausalEmbedding(TimeSeriesEmbedding):
    """A Causal model for embedding a time series
    
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
        self.model = CausalAutoencoder(
            self.n_latent,
            **kwargs
        )
    
    def fit(
        self, 
        X,
        y=None,
        learning_rate=1e-3, 
        batch_size=100, 
        train_steps=200,
        verbose=0
    ):
        """Fit the model with a time series X

        Parameters
        ----------
        X : array-like, shape (n_timepoints, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_timepoints, n_components)
            Transformed values.
        """
        # Make hankel matrix from dataset
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, len(Xs) - self.time_window, q=self.time_window)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
            loss="mse",
            #experimental_run_tf_function=False
        )    
        
        self.train_history = self.model.fit(
            x=tf.convert_to_tensor(X_train),                         
            y=tf.convert_to_tensor(X_train),
            epochs=train_steps,
            batch_size=batch_size,
            verbose=verbose
        )
             
    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), len(X) - self.time_window, q = self.time_window)
        X_new = self.model.encoder.predict(X_test)[self.time_window:]
        return X_new      
    


###------------------------------------###
#
#
#       Losses and Analysis Functions
#
#
###------------------------------------###

# @tf.function
# @tf.autograph.experimental.do_not_convert
def loss_false(code_batch, k=1):
    """
    An activity regularizer based on the False-Nearest-Neighbor
    Algorithm of Kennel, Brown,and Arbanel. Phys Rev A. 1992
    
    Parameters
    ----------
    code_batch: tensor
        (Batch size, Embedding Dimension) tensor of encoded inputs
    k: int 
        The number of nearest neighbors used to compute 
        neighborhoods.
    """

    _, n_latent = code_batch.get_shape()

    # changing these parameters is equivalent to changing the strength of the
    # regularizer, so we keep these fixed (these values correspond to the
    # original values used in Kennel et al 1992).
    rtol = 20.0
    atol = 2.0
    #     k_frac = 0.01
    #     n_batch = tf.cast(tf.keras.backend.shape(code_batch)[0], tf.float32)
    #     assert False, n_batch
    #     k = max(1, int(k_frac * n_batch))

    ## Vectorized version of distance matrix calculation
    tri_mask = tf.linalg.band_part(tf.ones((n_latent, n_latent), tf.float32), -1, 0)
    batch_masked = tf.multiply(tri_mask[:, tf.newaxis, :], code_batch[tf.newaxis, ...])
    X_sq = tf.reduce_sum(batch_masked * batch_masked, axis=2, keepdims=True)
    pdist_vector = (
        X_sq
        + tf.transpose(X_sq, [0, 2, 1])
        - 2 * tf.matmul(batch_masked, tf.transpose(batch_masked, [0, 2, 1]))
    )
    all_dists = pdist_vector
    all_ra = tf.sqrt(
        (1 / (tf.range(1, 1 + n_latent, dtype=tf.float32)))
        * tf.squeeze(
            tf.reduce_sum(
                tf.square(tf.math.reduce_std(batch_masked, axis=1, keepdims=True)),
                axis=2,
            )
        )
    )

    # Avoid singularity in the case of zeros
    all_dists = tf.clip_by_value(all_dists, 1e-14, tf.reduce_max(all_dists))

    # inds = tf.argsort(all_dists, axis=-1)
    _, inds = tf.math.top_k(-all_dists, k + 1)
    # top_k currently faster than argsort because it truncates matrix

    neighbor_dists_d = tf.gather(all_dists, inds, batch_dims=-1)
    neighbor_new_dists = tf.gather(all_dists[1:], inds[:-1], batch_dims=-1)

    # Eq. 4 of Kennel et al.
    scaled_dist = tf.sqrt(
        (tf.square(neighbor_new_dists) - tf.square(neighbor_dists_d[:-1]))
        / tf.square(neighbor_dists_d[:-1])
    )

    # Kennel condition #1
    is_false_change = scaled_dist > rtol
    # Kennel condition 2
    is_large_jump = neighbor_new_dists > atol * all_ra[:-1, tf.newaxis, tf.newaxis]

    is_false_neighbor = tf.math.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = tf.cast(is_false_neighbor, tf.int32)[..., 1 : (k + 1)]

    # Pad zero to match dimensionality of latent space
    reg_weights = 1 - tf.reduce_mean(
        tf.cast(total_false_neighbors, tf.float64), axis=(1, 2)
    )
    reg_weights = tf.pad(reg_weights, [[1, 0]])

    # Find average batch activity
    activations_batch_averaged = tf.sqrt(tf.reduce_mean(tf.square(code_batch), axis=0))

    # L2 Activity regularization
    activations_batch_averaged = tf.cast(activations_batch_averaged, tf.float64)
    loss = tf.reduce_sum(tf.multiply(reg_weights, activations_batch_averaged))

    return tf.cast(loss, tf.float32)


#@tf.function
def loss_cov(a, whiten=True):
    """
    The covariance loss, used to orthogonalize activations. Flattens the batch
    in order to compute elements of covariance matrix
    
    Parameters
    - a : B x N 
        Layer activations across a batch of size B
    - whiten : bool
        Whether to standardize the batch feature-wise
        
    Reference: Cogswell et al. ICLR 2016
    """
    a_mean = tf.reduce_mean(a, axis=0)
    n_batch = tf.cast(len(a), tf.float32)

    aw = (a - a_mean)
    if whiten:
        a_std = tf.math.reduce_std(a, axis=0)
        aw /= a_std

    cov = (1/n_batch)*tf.matmul(tf.transpose(aw), aw)

    loss = 0.5*(tf.square(tf.norm(cov)) - tf.square(tf.norm(tf.linalg.diag_part(cov))))
    
    return loss