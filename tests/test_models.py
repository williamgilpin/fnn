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

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKING_DIR)

import sys

sys.path.insert(1, os.path.join(WORKING_DIR, "fnn"))
from utils import hankel_matrix
from models import LSTMEmbedding, MLPEmbedding
from regularizers import FNN, loss_false


class TestUtilities(unittest.TestCase):
    """
    Tests helper utilities
    """

    def test_hankel(self):
        """
        Test hankel matrix construction
        """
        data_test = np.ones((1000, 3))
        hmat = hankel_matrix(data_test, p=100, q=3)

        assert hmat.shape == (
            100,
            3,
            3,
        ), "Hankel matrix construction failed, please check dependencies."


class TestModels(unittest.TestCase):
    """
    Tests models
    """

    def test_lstm(self):
        """
        Test initializing, compiling, and training an LSTM
        """
        model = LSTMEmbedding(2, time_window=5, random_state=0)
        model.fit(np.random.random(50))
        model.transform(np.random.random(50))
        latent = model.fit_transform(np.random.random(50))
        assert latent.shape == (45, 2), "Embedding has the wrong shape"

        # Try adding
        model = LSTMEmbedding(
            2, time_window=5, latent_regularizer=FNN(1e0), random_state=0
        )
        model.fit(np.random.random(50))
        model.transform(np.random.random(50))
        latent = model.fit_transform(np.random.random(50))
        assert latent.shape == (45, 2), "Embedding has the wrong shape"

    def test_mlp(self):
        """
        Test initializing, compiling, and training an MLP as a time-delay neural network
        """
        model = MLPEmbedding(2, time_window=5, random_state=0)
        model.fit(np.random.random(50))
        model.transform(np.random.random(50))
        latent = model.fit_transform(np.random.random(50))
        assert latent.shape == (45, 2), "Embedding has the wrong shape"

        # Try adding
        model = MLPEmbedding(
            2, time_window=5, latent_regularizer=FNN(1e0), random_state=0
        )
        model.fit(np.random.random(50))
        model.transform(np.random.random(50))
        latent = model.fit_transform(np.random.random(50))
        assert latent.shape == (45, 2), "Embedding has the wrong shape"

    def test_fnn_loss(self):
        """
        Test the false neighbors loss function
        """
        assert (
            loss_false(tf.zeros((25, 10)), k=3).numpy() < 1e-10
        ), "False neighbor loss did not return correct value."

    def test_fnn_regularizer(self):
        """
        Test false neighbors regularizer
        """
        fnn = FNN(1.0)
        example = tf.convert_to_tensor(np.random.random((10, 3)), tf.float32)
        assert np.isreal(
            fnn(example).numpy()
        ), "False neighbor regularizer did not return a real number."


if __name__ == "__main__":
    unittest.main()