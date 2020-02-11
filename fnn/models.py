"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf


def train_autoencoder(X_train, network_type='lstm', learning_rate=1e-3, lambda_latent=0.0, 
                      time_window=10, num_hidden=10, batch_size=100, random_seed=0, 
                      verbose=0, train_time=200):
    """
    This is a helper function that captures some of the boilerplate code for constructing
    and training an autoencoder (with default architecture)
    """
    
    tf.random.set_seed(random_seed)
    
    if network_type=='lstm':
        enc, dec = enc_dec_lstm(time_window, 1, num_hidden, 
                                rnn_opts={'activation': None, 
                                          'batch_size': batch_size})
    elif network_type=='mlp':
            enc, dec = enc_dec_tdnn(time_window, 1, num_hidden, 
                            rnn_opts={'activation': None, 
                                      'batch_size': batch_size})
    else:
        warnings.warn("Network type not recognized")

    inp = tf.keras.layers.Input(shape=(time_window, 1))
    code = enc(inp)
    reconstruction = dec(code)
    autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
    input_example = tf.cast(tf.convert_to_tensor(X_train[:batch_size]), tf.float32)
    reconstructed_example = autoencoder(input_example)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                        loss=loss_latent(code,batch_size, 
                                     lam_latent=lambda_latent),
                        experimental_run_tf_function=False)


    train_history = autoencoder.fit(x=tf.convert_to_tensor(X_train), 
                                    y=tf.convert_to_tensor(X_train),
                                    epochs=train_time,
                                    batch_size=batch_size,
                                    verbose=verbose)
    
    return enc, dec

    

def enc_dec_lstm(time_window, n_features, n_latent, hidden=[10], rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0)):
                #activation_func=tf.keras.activations.tanh):
    """
    Shallow LSTM autoencoder
    
    activation_func = tf.keras.activations.softplus
    """
    enc = tf.keras.Sequential()
    enc.add(tf.keras.layers.GaussianNoise(0.5)) # smooths the output

    enc.add(tf.keras.layers.LSTM(n_latent, input_shape=(time_window, n_features), 
                                      return_sequences=False, **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    #enc.add(tf.keras.layers.Activation(activation_func))
    

    dec = tf.keras.Sequential()
    #dec.add(tf.keras.layers.BatchNormalization())
    #dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.RepeatVector(time_window))
    dec.add(tf.keras.layers.GaussianNoise(0.5))
    dec.add(tf.keras.layers.LSTM(n_latent, return_sequences=True,  go_backwards=True,
                             **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    dec.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
    
    return enc, dec

def enc_dec_tdnn(time_window, n_features, n_latent, hidden=None, rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0)):
    """
    timedelay-NN (not recurrent)
    """
    if not hidden:
        hidden = [time_window, time_window]

    enc = tf.keras.Sequential()
    enc.add(tf.keras.layers.Flatten())
    enc.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(time_window,))) # smooths the output

    enc.add(tf.keras.layers.Dense(hidden[0], **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.Activation(activation_func))
    
    enc.add(tf.keras.layers.Dense(hidden[1], **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.Activation(activation_func))
    
    
    enc.add(tf.keras.layers.Dense(n_latent, input_shape=(time_window,), **rnn_opts))
    enc.add(tf.keras.layers.BatchNormalization())
    #enc.add(tf.keras.layers.Activation(activation_func))
    
    enc.add(tf.keras.layers.Reshape((n_latent,)))

    
    
    dec = tf.keras.Sequential()
    dec.add(tf.keras.layers.Flatten())
    dec.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(n_latent,)))
    
    dec.add(tf.keras.layers.Dense(hidden[1], input_shape=(n_latent,), **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    
    dec.add(tf.keras.layers.Dense(hidden[0], **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    
    
    dec.add(tf.keras.layers.Dense(n_latent, **rnn_opts))
    dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.Activation(activation_func))
    
    dec.add(tf.keras.layers.Reshape((time_window, n_features)))
    
    return enc, dec



###------------------------------------###
#
#
#       SCRATCH : Testing models and losses
#
#
###------------------------------------###

def loss_latent(latent, batch_size, lam_latent=1.0):
    """
    Build a custom loss function that includes layer terms, etc
    Does the covariance loss get summed over the batch?
    models : list of keras.Sequential() models
    """
    @tf.function
    def loss(y_true, y_pred):
        ## first term has shape (batch, lookback), do we really want to flatten it to just be (batch,)?
        ## can avoid by increasing dimensionality of last term. the grad wrt 
        total_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred, y_true), axis=1)
        total_loss += lam_latent*loss_false(latent, batch_size=batch_size)
        return total_loss

    return loss

###------------------------------------###
#
#
#       Losses and Analysis Functions
#
#
###------------------------------------###

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
        due to a cryptic bug
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
