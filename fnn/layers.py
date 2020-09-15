"""
TensorFlow functions to support the false nearest neighbor regularizer
"""
import tensorflow as tf
import warnings

class TimeMask(tf.keras.layers.Layer):
    """
    Apply a constant decay to the second dimension of a tensor
    Used for weighted more recent timepoints more than further back
    
    
    decay_rate : float
        The degree by which more recent values are favored
    
    reverse : bool
        Whether to reverse the mask direction across time
    """
    def __init__(self, decay_rate=0.0, reverse=True):
        super(TimeMask, self).__init__()
        self.decay_rate = decay_rate
        self.reverse = reverse
        #self.decay_mask = tf.keras.layers.Lambda(lambda x: x[:, :-chomp_size, :])
        
    def build(self, input_shape):
        self.n_batch, self.n_time, self.n_features = input_shape
        indices = tf.cast(tf.linspace(0, 1, self.n_time), tf.float32)
        self.mask = tf.math.exp(-self.decay_rate * indices)[:, None]
        if self.reverse:
            self.mask = self.mask[::-1]
        
    def call(self, x):
        return self.mask * x

class Chomp1d(tf.keras.layers.Layer):
    """
    Removes the last elements of a time series. Assumes that the time series
    has shape (B, L, C), where B is batch size, L is length of input, and C
    is the number of input channels
    
    Returns a tensor of shape (B, L - s, C), where s is the chomp size parameter
    
    See
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp1 = tf.keras.layers.Lambda(lambda x: x[:, :-chomp_size, :])

    def call(self, x):
        return self.chomp1(x)

class CausalConv1D(tf.keras.layers.Layer):
    """
    Implement causal convolutions using 1x1 filters. Useful for bypassing
    issues with tensorflow's native Conv1D 

    transpose : bool
        Whether to use transposed (upsampling) convolutions
    """
    def __init__(self, 
                 output_dim,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 transpose=False,
                 **kwargs
                ):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.chunk = (kernel_size - 1)*dilation_rate + 1
        self.chomp1 = Chomp1d(self.chunk - 1)
        self.is_transpose = transpose
        if transpose:
            self.conv1 = tf.keras.layers.Conv1DTranspose(output_dim, 
                                        kernel_size=1,
                                        #output_padding=0, 
                                        strides=strides, 
                                        dilation_rate=1, 
                                        **kwargs)
        else:
            self.conv1 = tf.keras.layers.Conv1D(output_dim, 
                                        kernel_size=1, 
                                        strides=strides, 
                                        dilation_rate=1, 
                                        **kwargs)
        
    
    def build(self, input_shape):
        #crop_size = input_shape[0] // self.chunk
        self.n_batch, self.n_time, self.n_features = input_shape
        n_features = self.n_features

        n = self.kernel_size
        filters = [i*self.dilation_rate*[0] + [1] + (n-i-1)*self.dilation_rate*[0] for i in range(n)]
        filters = tf.convert_to_tensor(filters)
        pad_zeros = [tf.cast(tf.zeros(filters.shape), tf.int32)]
        all_filters = list()
        for i in range(n_features):
            filter_layer = i*pad_zeros + [filters] + (n_features - i - 1)*pad_zeros
            filter_layer = tf.transpose(tf.convert_to_tensor(filter_layer), perm=(1,0,2))
            all_filters.append(filter_layer)
        all_filters = tf.concat(all_filters, axis=0)
        self.filters = tf.transpose(tf.cast(all_filters, tf.float32))
        
    def call(self, x):
        x = tf.pad(x, [[0, 0], [self.chunk-1, self.chunk-1], [0,0]], "CONSTANT")
        y = tf.nn.conv1d(x, self.filters, stride=1, padding='VALID')
        y = self.chomp1(y)
        h = self.conv1(y)
        return h
    
class CausalEncoder(tf.keras.Model):
    """
    """
    def __init__(self,
                 n_latent,
                 network_shape=[10,10],
                 kernel_size=3,
                 strides=1,
                 dilation_scale=1,
                 latent_regularizer=None,
                 activation_func=tf.nn.elu,
                 rnn_opts={}
                ):
        super(CausalEncoder, self).__init__()
        self.gn = tf.keras.layers.GaussianNoise(0.5)
        causal_steps = []
        for i, hidden_size in enumerate(network_shape):
            causal_steps.append(
                CausalConv1D(
                    hidden_size, 
                    kernel_size,
                    strides=strides, 
                    dilation_rate=dilation_scale**i,
                    activation=None, 
                    name="conv" + str(i),
                    **rnn_opts
                )
            )
            causal_steps.append(tf.keras.layers.BatchNormalization())
            causal_steps.append(tf.keras.layers.Activation(activation_func))
            causal_steps.append(TimeMask(20.0, reverse=True))
            
        self.causal_block = tf.keras.Sequential(causal_steps)
        self.skip_down = tf.keras.layers.Conv1D(
            network_shape[-1], 
            kernel_size,
            strides=strides**len(network_shape),
            padding="SAME",
            activation=None, 
            name="skip_down",
            **rnn_opts
        )
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            n_latent,
            activation=None,
            activity_regularizer=latent_regularizer
        )
        
    def call(self, x):
        xn = self.gn(x)
        y = self.dense1(self.flat1(self.causal_block(xn) + self.skip_down(xn)))
#         y = self.dense1(self.flat1(self.causal_block(xn)))
        return y
        
        
    
class CausalBlock(tf.keras.layers.Layer):
    """
    Implement a residual causal convolution block 
    
    final : bool
        whether to avoid activation of residual transform 
    
    "Unsupervised Scalable Representation Learning for Multivariate Time Series"
    Franceschi, Dieuleveut, Jaggi. NeurIPS 2019
    
    
    DEV : Add BatchNorm or Weightnorm after each convolution
    """
    def __init__(self, 
                 out_channels, 
                 kernel_size, 
                 dilation_rate=1,
                 transpose=False,
                 activation=tf.nn.elu,
                 final=False,
                 strides=1,
                 **kwargs):
        super(CausalBlock, self).__init__()
        self.activation = activation
        self.out_channels = out_channels
        self.strides = strides
        self.cause1 = CausalConv1D(out_channels, 
                                   kernel_size = kernel_size, 
                                   dilation_rate = dilation_rate,
                                   strides = strides,
                                   transpose=transpose,
                                  **kwargs)
        self.cause2 = CausalConv1D(out_channels, 
                           kernel_size = kernel_size, 
                           dilation_rate = dilation_rate,
                           strides = strides,
                           transpose=transpose,
                                  **kwargs)
        # Skip connection
        if transpose:
            self.rescale = tf.keras.layers.Conv1DTranspose(out_channels, 
                                kernel_size=1, 
                                strides = strides**2,
                                **kwargs)          
        else:
            self.rescale = tf.keras.layers.Conv1D(out_channels, 
                                kernel_size=1, 
                                strides = strides**2,
                                **kwargs)
        
        if final:
            self.final_activation = tf.keras.layers.Layer() # Identity layer
        else:
            self.final_activation = activation
            
    def build(self, input_shape):
        self.n_batch, self.n_time, self.n_features = input_shape 
        self.do_rescale = (self.n_features != self.out_channels) or (self.strides > 1)

            
    def call(self, x):
        y = self.activation(self.cause1(x))
        y = self.activation(self.cause2(y))
        if self.do_rescale:
            return self.final_activation(y + self.rescale(x))
        else:
            return self.final_activation(y + x)
 
    
class CausalCNN(tf.keras.layers.Layer):
    """
    Implement a deep causal convolutional network 
    
    channels : int or list
        The number of hidden units, or a list of units for hidden layers
    
    "Unsupervised Scalable Representation Learning for Multivariate Time Series"
    Franceschi, Dieuleveut, Jaggi. NeurIPS 2019
    """
    def __init__(self, 
                 out_channels,
                 channels, 
                 kernel_size=3,
                 strides=1,
                 transpose=False,
                 activation=tf.nn.elu,
                 **kwargs):
        super(CausalCNN, self).__init__()
        
        # check that the specified depth and network shape are the same
        if tf.rank(channels) == 0:
            channels = [channels]
        
        layers = []
        for i, n_channel in enumerate(channels):

            layers += [CausalBlock(
                n_channel, 
                kernel_size, 
                dilation_rate=2**i,
                strides=strides,
                #strides=(1 + kernel_size // 2), # causes downsampling of time series
                transpose=transpose,
                activation=activation,
                final=False,
                **kwargs
            )]

        # Final layer
        layers += [CausalBlock(
            out_channels, 
            kernel_size, 
            dilation_rate=1,
            transpose=transpose,
            activation=activation,
            final=True,
            **kwargs
        )]
        
        self.network = tf.keras.Sequential(layers)
        
    def call(self, x):
        return self.network(x)

# class CausalEncoder(tf.keras.layers.Layer):
#     """
#     Implement an encoder that converts a (B, T, C) time series to (B, L)
    
#     channels : int or list
#         The number of hidden units, or a list of units for hidden layers
    
#     "Unsupervised Scalable Representation Learning for Multivariate Time Series"
#     Franceschi, Dieuleveut, Jaggi. NeurIPS 2019
#     """
#     def __init__(self, 
#                  out_channels,
#                  channels, 
#                  kernel_size=3,
#                  activation=tf.nn.elu,
#                  **kwargs):
#         super(CausalCNN, self).__init__()
        
#         # check that the specified depth and network shape are the same
#         if tf.rank(channels) == 0:
#             channels = [channels]
        
#         layers = []
#         for i, n_channel in enumerate(channels):

#             layers += [CausalBlock(
#                 n_channel, 
#                 kernel_size, 
#                 dilation_rate=2**i,
#                 strides=(1 + kernel_size // 2), # causes downsampling of time series
#                 activation=activation,
#                 final=False,
#                 **kwargs
#             )]

#         # Final layer
#         layers += [
#             tf.keras.layers.Reshape([-1]),
#             tf.keras.layers.Dense(
#                 out_channels, 
#                 activation=activation
#                 )
#             ]
        
#         self.network = tf.keras.Sequential(layers)
        
#     def call(self, x):
#         return self.network(x)

# class CausalDecoder(tf.keras.layers.Layer):
#     """
#     Implement an decoder that converts a (B, L) time series to (B, T, C)
    
#     channels : int or list
#         The number of hidden units, or a list of units for hidden layers
    
#     "Unsupervised Scalable Representation Learning for Multivariate Time Series"
#     Franceschi, Dieuleveut, Jaggi. NeurIPS 2019
#     """
#     def __init__(self, 
#                  out_channels,
#                  channels, 
#                  kernel_size=3,
#                  activation=tf.nn.elu,
#                  **kwargs):
#         super(CausalCNN, self).__init__()
        
#         # check that the specified depth and network shape are the same
#         if tf.rank(channels) == 0:
#             channels = [channels]
        
#         layers = [tf.keras.layers.Reshape((time_window, n_features))]
#         for i, n_channel in enumerate(channels):

#             layers += [CausalBlock(
#                 n_channel, 
#                 kernel_size, 
#                 dilation_rate=2**i,
#                 strides=(1 + kernel_size // 2), # causes downsampling of time series
#                 activation=activation,
#                 final=False,
#                 **kwargs
#             )]

#         # Final layer
#         layers += [
#             tf.keras.layers.Reshape([-1]),
#             tf.keras.layers.Dense(
#                 out_channels, 
#                 activation=activation
#                 )
#             ]
        
#         self.network = tf.keras.Sequential(layers)
        
#     def call(self, x):
#         return self.network(x)

# class CausalEncodingLayer(tf.keras.layers.Layer):
#     """
#     Downsample a time series with a skip connection
#     Keyword arguments get passed to the trainable layers
#     Input shape: (None, T, D)
#     Output shape: (None, L)
#     """
#     def __init__(self, 
#                  output_dim, 
#                  stride=3,
#                  activation=tf.nn.elu, 
#                  **kwargs):
#         super(CausalEncodingLayer, self).__init__()
#         self.output_dim = output_dim
#         self.kwargs = kwargs
#         self.activation = activation
#         self.stride = stride
        
#         # layers
#         # self.pool1 = tf.keras.layers.Lambda(lambda x: x[:, ::self.stride, :])
#         self.pool1 = tf.keras.layers.AveragePooling1D(pool_size=self.stride, padding="valid")
#         self.flat1 = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(self.output_dim, **kwargs)
#         self.dense2 = tf.keras.layers.Dense(self.output_dim, **kwargs)
#         self.flat2 = tf.keras.layers.Flatten()
#         self.skip1 = tf.keras.layers.Dense(self.output_dim, **kwargs)
        
#     def call(self, input_layer):
#         h1 = self.pool1(input_layer)
#         h1_flat = self.flat1(h1)
#         h1 = self.activation(self.dense1(h1_flat))
#         h2 = self.dense2(h1)
        
#         if self.stride > 1:
#             flat_in = self.flat2(input_layer)
#             skip = self.skip1(flat_in) # no activation
#             h2 = h2 + skip
#         return h2
    
# class CausalDecodingLayer(tf.keras.layers.Layer):
#     """
#     Upsample a time series with a skip connection
#     Keyword arguments get passed to the trainable layers
#     Input shape: (None, L)
#     Output shape: (None, T, D)
#     """
#     def __init__(self, 
#                  output_dim, 
#                  output_channels=1,
#                  stride=3,
#                  activation=tf.nn.elu, 
#                  **kwargs):
#         super(CausalDecodingLayer, self).__init__()
#         self.output_dim = output_dim
#         self.output_channels = output_channels,
#         self.total_dim = output_dim*output_channels
#         self.kwargs = kwargs
#         self.activation = activation
#         self.stride = stride
        
#         # layers
#         # self.pool1 = tf.keras.layers.Lambda(lambda x: x[:, ::self.stride, :])
#         self.paddim = tf.keras.layers.Lambda(lambda x: x[..., None])
#         self.up1 = tf.keras.layers.UpSampling1D(size=self.stride)
#         self.dense1 = tf.keras.layers.Dense(self.total_dim, **kwargs)
#         self.flat1 = tf.keras.layers.Flatten()
#         self.reshape1 = tf.keras.layers.Reshape((output_dim, output_channels))
        
        
#         self.dense2 = tf.keras.layers.Dense(self.total_dim, **kwargs)
    
#     def call(self, input_layer):
        
#         h1 = self.up1(self.paddim(input_layer))
#         h2_flat = self.flat1(h1)
#         h2 = self.activation(self.dense1(h2_flat))
#         if self.stride > 1:
#             skip = self.dense2(input_layer) # no activation
#             h2 = h2 + skip
#         h2 = self.reshape1(h2)   
#         return h2
