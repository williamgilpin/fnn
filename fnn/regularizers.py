import warnings
import tensorflow as tf

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

    def __init__(self, strength, k=1):
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
    
# Alias of built-in for API consistency
L1Reg = tf.keras.regularizers.L1 
L2Reg = tf.keras.regularizers.L2


###------------------------------------###
#
#
#       Internal and Utility Functions
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
    _, inds = tf.math.top_k(-all_dists, int(k + 1))
    # top_k currently faster than argsort because it truncates matrix

    neighbor_dists_d = tf.gather(all_dists, inds, batch_dims=-1)
    neighbor_new_dists = tf.gather(all_dists[1:], inds[:-1], batch_dims=-1)

    # Eq. 4 of Kennel et al.
    scaled_dist = tf.sqrt(
        (neighbor_new_dists - neighbor_dists_d[:-1]) / neighbor_dists_d[:-1]
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
def loss_cov(a, whiten=False):
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
    n_batch = tf.cast(tf.shape(a)[0], tf.float32)

    aw = (a - a_mean)
    if whiten:
        a_std = tf.math.reduce_std(a, axis=0)
        aw /= a_std

    cov = (1/n_batch)*tf.matmul(tf.transpose(aw), aw)

    loss = 0.5*(tf.square(tf.norm(cov)) - tf.square(tf.norm(tf.linalg.diag_part(cov))))
    
    return loss