"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf

def train_autoencoder(X_train, network_type='lstm',
                      time_window=10, num_hidden=10, n_features=1, network_shape=None,
                      lambda_latent=0.0, lambda_ortho=0.0, learning_rate=1e-3, batch_size=100, train_time=200,
                      verbose=0, random_seed=0, return_history=False):
    """
    This is a helper function that captures some of the boilerplate code for constructing
    and training an autoencoder (with default architecture).
    
    Parameters
    ----------
    - network_type : 'lstm' or 'mlp'
        Whether to use an LSTM or a feedforward network (equiv. to a time-delay neural network)
    - n_features : int
        The dimensionality of a point in the time series
    - network_shape : int or list
        The width of hidden layers
    - time_window : int
        The number of past timesteps to use for embedding
    - num_hidden : int
        The number of latent variables
    - lambda_latent : float
        The relative weight of the false-neighbors loss during training
    - lambda_ortho : float
        The relative weight of the orthogonality penalty during training
    - learning_rate : float
        The learning rate to use for training
    - batch_size : int
        The number of samples in each training batch
    - train_time : int
        The number of training epochs
    - verbose : int
        The amount of detail to record during training
    - random_seed : int
        The value for random initialization of the network
    - return_history : bool
        Whether to return the training history with the trained models
    """
    
    tf.random.set_seed(random_seed)
    
    if not network_shape:
        network_shape = int(time_window)
    if type(network_shape) is not list:
        if network_type == 'lstm':
            network_shape = [network_shape]
        elif network_type == 'mlp':
            network_shape = [network_shape, network_shape]
        else:
            pass
    if network_type == 'lstm':
        enc, dec = enc_dec_lstm(time_window, n_features, num_hidden, 
                                hidden = network_shape,
                                rnn_opts={'activation': None, 
                                          'batch_size': batch_size})
    elif network_type == 'mlp':
            enc, dec = enc_dec_tdnn(time_window, n_features, num_hidden, 
                                    hidden = network_shape,
                            rnn_opts={'activation': None, 
                                      'batch_size': batch_size})
    else:
        warnings.warn("Network type not recognized")

    inp = tf.keras.layers.Input(shape=(time_window, n_features))
    code = enc(inp)
    reconstruction = dec(code)
    autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
    #input_example = tf.cast(tf.convert_to_tensor(X_train[:batch_size]), tf.float32)
    #reconstructed_example = autoencoder(input_example)
    
    if lambda_latent == 0 and lambda_ortho == 0:
        loss_term = "mse"
    elif lambda_latent == 0 and lambda_ortho > 0:
        loss_term = loss_ortho(code, lam=lambda_ortho)
    else:
        loss_term = loss_latent(code, batch_size, lam=lambda_latent)
        
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
        loss=loss_term,
        metrics=[mse_loss],
        experimental_run_tf_function=False
    )    

    train_history = autoencoder.fit(x=tf.convert_to_tensor(X_train), 
                                    y=tf.convert_to_tensor(X_train),
                                    epochs=train_time,
                                    batch_size=batch_size,
                                    verbose=verbose)
    if return_history:
        return enc, dec, train_history
    
    return enc, dec

    

def enc_dec_lstm(time_window, n_features, n_latent, hidden=[10], rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0)):
                #activation_func=tf.keras.activations.tanh):
    """
    Build a one-layer LSTM autoencoder
    
    Parameters
    ----------
    - activation_func : function
        The nonlinearity to apply to all layers. Try tf.keras.activations.relu
        or tf.keras.activations.softplus
    """
    enc = tf.keras.Sequential()
    enc.add(tf.keras.layers.GaussianNoise(0.5))  # smooths the output

    enc.add(
        tf.keras.layers.LSTM(
            n_latent,
            input_shape=(time_window, n_features),
            return_sequences=False,
            **rnn_opts
        )
    )
    enc.add(tf.keras.layers.BatchNormalization())
    # enc.add(tf.keras.layers.Activation(activation_func))

    ## Latent ##

    dec = tf.keras.Sequential()
    # dec.add(tf.keras.layers.BatchNormalization())
    # dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.RepeatVector(time_window))
    dec.add(tf.keras.layers.GaussianNoise(0.5))
    dec.add(
        tf.keras.layers.LSTM(
            n_latent, return_sequences=True, go_backwards=True, **rnn_opts
        )
    )
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
    
    return enc, dec

def enc_dec_tdnn(time_window, n_features, n_latent, hidden=None, rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0)):
    """
    Build a time-delay neural network (a non-stateful architecture)
    """
    if not hidden:
        hidden = [time_window, time_window]

    enc = tf.keras.Sequential()
    enc.add(tf.keras.layers.Flatten())
    enc.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(time_window,))) # smooths the output
    
    for hidden_unit in hidden:
        enc.add(tf.keras.layers.Dense(hidden_unit, **rnn_opts))
        enc.add(tf.keras.layers.BatchNormalization())
        enc.add(tf.keras.layers.Activation(activation_func))
    
    enc.add(tf.keras.layers.Dense(n_latent, input_shape=(time_window,), **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    #enc.add(tf.keras.layers.Activation(activation_func))
    
    enc.add(tf.keras.layers.Reshape((n_latent,)))

    ## Latent ##
    
    dec = tf.keras.Sequential()
    dec.add(tf.keras.layers.Flatten())
    dec.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(n_latent,)))
    
    for hidden_unit in hidden[::-1]:
        dec.add(tf.keras.layers.Dense(hidden_unit,  **rnn_opts))
        dec.add(tf.keras.layers.BatchNormalization())
        dec.add(tf.keras.layers.Activation(activation_func))
    
    
    dec.add(tf.keras.layers.Dense(time_window*n_features, **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    
    dec.add(tf.keras.layers.Reshape((time_window, n_features)))
    
    return enc, dec

def enc_dec_rnn(time_window, n_features, n_latent, hidden=None, rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0)):
    """
    Build a two-layer vanilla RNN autoencoder
    """

    if not hidden:
        hidden = [time_window]

    enc = tf.keras.Sequential()
    enc.add(tf.keras.layers.GaussianNoise(0.5)) # smooths the output
    
    enc.add(tf.keras.layers.SimpleRNN(hidden[0], 
                                      input_shape=(time_window, n_features), 
                                      return_sequences=True, 
                                      **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.Activation(activation_func))
    
    enc.add(tf.keras.layers.SimpleRNN(n_latent, 
                                      return_sequences=False, 
                                      **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())


    dec = tf.keras.Sequential()
    dec.add(tf.keras.layers.RepeatVector(time_window))
    dec.add(tf.keras.layers.GaussianNoise(0.5))
    dec.add(tf.keras.layers.SimpleRNN(n_latent, 
                                      return_sequences=True,  
                                      go_backwards=True,
                                      **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.SimpleRNN(hidden[0], 
                                      return_sequences=True,  
                                      go_backwards=True,
                                      **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
    
    return enc, dec

###------------------------------------###
#
#
#       Losses and Analysis Functions
#
#
###------------------------------------###

def loss_latent(latent, batch_size, lam=1.0):
    """
    Build a custom loss function that keras.compile will accept.
    
    Parameters
    ----------
    - latent : B x T x L)
        A batch of latent activations
    - batch_size : int
        The expected batch size
    - lam : float
        The relative weight of the fnn regularizer. Expressed relative to the
        standard autoencoder reconstruction loss, which has constant weight 1.0
        
    Returns
    -------
    - loss : function
        A keras-friendly loss function that takes two batches of labels as arguments
    """
    @tf.function
    def loss(y_true, y_pred):
        """Loss function generated automatically by loss_latent()"""
        total_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred, y_true), axis=1)
        total_loss += lam*loss_false(latent, batch_size=batch_size)
        return total_loss

    return loss

def loss_ortho(latent, lam=1.0):
    """
    Build a custom loss function that keras.compile will accept.
    
    Parameters
    ----------
    - latent : B x T x L)
        A batch of latent activations
    - batch_size : int
        The expected batch size
    - lam_latent : float
        The relative weight of the regularizer. Expressed relative to the
        standard autoencoder reconstruction loss, which has constant weight 1.0    
   
   Returns
    -------
    - loss : function
        A keras-friendly loss function that takes two batches of labels as arguments
    """
    @tf.function
    def loss(y_true, y_pred):
        ## first term has shape (batch, lookback), do we really want to flatten it to just be (batch,)?
        ## can avoid by increasing dimensionality of last term. the grad wrt 
        total_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred, y_true), axis=1)
        total_loss += lam*loss_cov(latent)
        return total_loss

    return loss


@tf.function
def loss_false(code_batch, batch_size=1, k=None):
    """
    An activity regularizer based on the False-Nearest-Neighbor
    Algorithm of Kennel, Brown,and Arbanel. Phys Rev A. 1992
    
    Parameters
    ----------
    - code_batch : B x L
        The encoded inputs, with dimensionality L
    - batch_size : int
        The batch size. For some reason, this has to be passed manually
        due to a bug on keras' end
    - k : int 
        DEPRECATED. The number of nearest neighbors to use to compute 
        neighborhoods. Right now this is set to a constant, since it doesn't 
        really affect the embedding

    Development
    -----------
    + Keras currently struggles to find n_batch automatically, and so it has
    to be passed as a kwarg. Hopefully this will get fixed in a future release
    + Try activity regularizing based on the variance of a neuron, rather than
    its absolute average activity
    
    """
    ## Uncomment these lines, and comment out the next line,
    ## if the batch_size bug is present in your version of keras
    _, n_latent = code_batch.get_shape()
    n_batch = batch_size

    # changing these parameters is equivalent to
    # changing the strength of the regularizer, so we keep these fixed (these values
    # correspond to the original values used in Kennel et al 1992).
    rtol = 20.0
    atol = 2.0
    k_frac = .01

    k = max(1, int(k_frac*n_batch))

    ## Vectorized version of distance matrix calculation
    tri_mask = tf.linalg.band_part(tf.ones((n_latent, n_latent), tf.float32), -1, 0)
    batch_masked = tf.multiply(tri_mask[:, tf.newaxis, :], code_batch[tf.newaxis, ...])
    X_sq = tf.reduce_sum(batch_masked*batch_masked, axis=2, keepdims=True)
    pdist_vector = X_sq + tf.transpose(X_sq, [0,2,1]) - 2*tf.matmul(batch_masked, tf.transpose(batch_masked, [0,2,1])) 
    all_dists = pdist_vector
    all_ra = tf.sqrt((1/(n_batch*tf.range(1, 1+n_latent, dtype=tf.float32)))*tf.reduce_sum(tf.square(batch_masked - tf.reduce_mean(batch_masked, axis=1, keepdims=True)), axis=(1,2)))
    
    # Avoid singularity in the case of zeros
    all_dists = tf.clip_by_value(all_dists, 1e-14, tf.reduce_max(all_dists))

    #inds = tf.argsort(all_dists, axis=-1)
    _, inds = tf.math.top_k(-all_dists, k+1)
    # top_k currently faster than argsort because it truncates matrix
    
    neighbor_dists_d = tf.gather(all_dists, inds, batch_dims=-1)
    neighbor_new_dists = tf.gather(all_dists[1:], inds[:-1], batch_dims=-1)
    
    # Eq. 4 of Kennel et al.
    scaled_dist = tf.sqrt((tf.square(neighbor_new_dists) - tf.square(neighbor_dists_d[:-1]))/tf.square(neighbor_dists_d[:-1]))
    
    # Kennel condition #1
    is_false_change = (scaled_dist > rtol) 
    # Kennel condition 2
    is_large_jump = (neighbor_new_dists > atol*all_ra[:-1, tf.newaxis, tf.newaxis])

    is_false_neighbor = tf.math.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = tf.cast(is_false_neighbor, tf.int32)[..., 1:(k+1)]
    
    # Pad zero to match dimensionality of latent space
    reg_weights = 1 - tf.reduce_mean(tf.cast(total_false_neighbors, tf.float64), axis=(1,2))
    reg_weights = tf.pad(reg_weights, [[1, 0]])

    # Find batch average activity
    activations_batch_averaged  = tf.sqrt(tf.reduce_mean(tf.square(code_batch), axis=0))

    # L2 Activity regularization
    activations_batch_averaged = tf.cast(activations_batch_averaged, tf.float64)
    loss = tf.reduce_sum(tf.multiply(reg_weights, activations_batch_averaged))
    
    return tf.cast(loss, tf.float32)

def mse_loss(y_true, y_pred):
    """
    The mean squared error loss between observed and true labels
    """
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred, y_true), axis=1)

@tf.function
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