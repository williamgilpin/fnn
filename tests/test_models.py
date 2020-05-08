"""
Test the models and regularizer
> python test_models.py

TODO: Add tests for non-nn models
"""
#!/usr/bin/env python
import os, glob
import numpy as np
import unittest

import tensorflow as tf

WORKING_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
print(WORKING_DIR)

import sys
sys.path.insert(1, os.path.join(WORKING_DIR, 'fnn'))
from utils import hankel_matrix
from models import enc_dec_lstm, enc_dec_tdnn, loss_false, loss_latent



class TestUtilities(unittest.TestCase):
    """
    Tests helper utilities
    """
    
    def test_hankel(self):
        """
        Test hankel matrix construction
        """
        data_test = np.ones((1000, 3))
        hmat = hankel_matrix(data_test, 100, q=3)
        
        assert (
            hmat.shape == (100, 3, 3)
        ), "Hankel matrix construction failed, please check dependencies."

class TestModels(unittest.TestCase):
    """
    Tests models
    """
    
    def test_lstm(self):
        """
        Test initializing, compiling, and training an LSTM
        """
        enc, dec = enc_dec_lstm(10, 2, 5,  # time_window, n_features, n_latent
                        rnn_opts={'activation': None, 
                                  'batch_size': 25})
        input_val = tf.ones((25, 10, 2))
        
        latent = enc(input_val)
        assert (
            latent.numpy().shape == (25, 5)
        ), "Latent variables have wrong shape"
        
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Reconstruction has wrong shape"
        
        
        inp = tf.keras.layers.Input(shape=(10, 2))
        code = enc(inp)
        reconstruction = dec(code)
        autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), 
                            loss='mse')
        
        input_val = tf.ones((25, 10, 2))
        output = autoencoder(input_val)
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Compiled model has wrong shape"
        
        X_train = tf.ones((300, 10, 2))
        autoencoder.fit(x=tf.convert_to_tensor(X_train), 
                                    y=tf.convert_to_tensor(X_train),
                                    epochs=5,
                                    batch_size=25,
                                    verbose=0)
        input_val = tf.ones((25, 10, 2))
        output = autoencoder(input_val)
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Trained model has wrong shape"
        
    def test_mlp(self):
        """
        Test initializing, compiling, and training an MLP as a time-delay neural network
        """
        enc, dec = enc_dec_tdnn(10, 2, 5,  # time_window, n_features, n_latent
                        rnn_opts={'activation': None, 
                                  'batch_size': 25})
        input_val = tf.ones((25, 10, 2))
        
        latent = enc(input_val)
        assert (
            latent.numpy().shape == (25, 5)
        ), "Latent variables have wrong shape"
        
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Reconstruction has wrong shape"
        
        
        inp = tf.keras.layers.Input(shape=(10, 2))
        code = enc(inp)
        reconstruction = dec(code)
        autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), 
                            loss='mse')
        
        input_val = tf.ones((25, 10, 2))
        output = autoencoder(input_val)
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Compiled model has wrong shape"
        
        X_train = tf.ones((300, 10, 2))
        autoencoder.fit(x=tf.convert_to_tensor(X_train), 
                                    y=tf.convert_to_tensor(X_train),
                                    epochs=5,
                                    batch_size=25,
                                    verbose=0)
        input_val = tf.ones((25, 10, 2))
        output = autoencoder(input_val)
        output = dec(enc(input_val))
        assert (
            output.numpy().shape == input_val.numpy().shape
        ), "Trained model has wrong shape"
        
    
    def test_fnn_loss(self):
        """
        Test the false neighbors loss function
        """
        assert (
            loss_false(tf.zeros((25, 10)), 25).numpy() < 1e-10
        ), "False neighbor loss is working"
        
    def test_latent_loss(self):
        """
        Test the combined reconstruction and false neighbors loss function
        """
        f = loss_latent(tf.zeros((25, 10)), 25)
        assert (
            tf.reduce_sum(f(tf.zeros((25, 10, 1)), 
                            tf.zeros((25, 10, 1)))).numpy() < 1e-10
        ), "Combined false-neighbor and reconstruction loss is working"
        

if __name__ == "__main__":
    unittest.main()