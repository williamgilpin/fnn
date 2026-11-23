#!/usr/bin/env python
"""Unit tests for the fnn regularizers, embedding models, and utilities.

Run with:
    uv run python tests/test_models.py
or:
    uv run pytest
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

from fnn.models import (
    AMIEmbedding,
    CausalEmbedding,
    ConstantLagEmbedding,
    ETDConstrainedEmbedding,
    ETDEmbedding,
    LSTMEmbedding,
    MLPEmbedding,
    TICAEmbedding,
)
from fnn.regularizers import FNN, DeCov, L1Reg, L2Reg, loss_cov, loss_false
from fnn.utils import hankel_matrix, resample_dataset, standardize_ts


class TestUtilities(unittest.TestCase):
    """Tests delay-embedding and preprocessing utilities."""

    def test_hankel(self) -> None:
        data_test = np.ones((1000, 3))
        hmat = hankel_matrix(data_test, p=100, q=3)
        self.assertEqual(hmat.shape, (100, 3, 3))

    def test_standardize_ts(self) -> None:
        arr = np.column_stack([np.arange(100, dtype=float), np.ones(100)])
        std_arr = standardize_ts(arr)
        self.assertAlmostEqual(float(np.mean(std_arr[:, 0])), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(std_arr[:, 0])), 1.0, places=6)
        self.assertTrue(np.allclose(std_arr[:, 1], 0.0))

    def test_resample_dataset(self) -> None:
        data = np.arange(100).reshape(50, 2)
        idx, sub = resample_dataset(data, n_samples=10, random_state=0)
        self.assertEqual(idx.shape, (10,))
        self.assertEqual(sub.shape, (10, 2))
        self.assertTrue(np.array_equal(sub, data[idx]))


class TestRegularizers(unittest.TestCase):
    """Tests FNN and auxiliary latent regularizers."""

    def test_fnn_loss_zeros(self) -> None:
        z = torch.zeros((25, 10), requires_grad=True)
        val = loss_false(z, k=3)
        self.assertLess(val.item(), 1e-10)
        val.backward()
        assert z.grad is not None
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_fnn_regularizer_gradients(self) -> None:
        torch.manual_seed(0)
        fnn = FNN(1.0)
        z = torch.randn((64, 6), requires_grad=True)
        val = fnn(z)
        self.assertEqual(val.ndim, 0)
        self.assertGreater(val.item(), 0.0)
        val.backward()
        assert z.grad is not None
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.allclose(z.grad[:, 0], torch.zeros(64)))
        self.assertGreater(z.grad[:, 1:].abs().sum().item(), 0.0)

    def test_fnn_edge_cases(self) -> None:
        self.assertEqual(loss_false(torch.randn(10, 1)).item(), 0.0)
        self.assertEqual(loss_false(torch.randn(1, 5)).item(), 0.0)
        z64 = torch.randn((32, 5), dtype=torch.float64, requires_grad=True)
        val64 = loss_false(z64)
        self.assertEqual(val64.dtype, torch.float64)
        val64.backward()
        self.assertTrue(torch.isfinite(z64.grad).all())

    def test_other_regularizers(self) -> None:
        torch.manual_seed(0)
        z = torch.randn((32, 4), requires_grad=True)
        for reg in (DeCov(1.0), DeCov(1.0, whiten=True), L1Reg(0.1), L2Reg(0.1)):
            z.grad = None
            val = reg(z)
            self.assertGreaterEqual(val.item(), 0.0)
            val.backward()
            assert z.grad is not None
            self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreaterEqual(loss_cov(z.detach()).item(), 0.0)

    def test_custom_pytorch_model(self) -> None:
        torch.manual_seed(0)
        encoder = nn.Sequential(nn.Linear(10, 16), nn.ELU(), nn.Linear(16, 6))
        decoder = nn.Sequential(nn.Linear(6, 16), nn.ELU(), nn.Linear(16, 10))
        fnn_reg = FNN(1e-2)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2
        )

        t = torch.linspace(0, 20, 128)
        proj = torch.randn(2, 10)
        x = torch.stack([torch.sin(t), torch.cos(t)], dim=1) @ proj

        losses = []
        for _ in range(25):
            z = encoder(x)
            x_hat = decoder(z)
            loss = F.mse_loss(x_hat, x) + fnn_reg(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0])


class TestModels(unittest.TestCase):
    """Tests neural and baseline time-series embedding models."""

    def test_lstm(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random(50)
        model = LSTMEmbedding(2, time_window=5, random_state=0)
        model.fit(x, train_steps=3)
        self.assertEqual(model.transform(x).shape, (45, 2))
        latent = model.fit_transform(x, train_steps=3)
        self.assertEqual(latent.shape, (45, 2))

        model = LSTMEmbedding(
            2, time_window=5, latent_regularizer=FNN(1.0), random_state=0
        )
        latent = model.fit_transform(x, train_steps=3)
        self.assertEqual(latent.shape, (45, 2))
        self.assertEqual(len(model.train_history["loss"]), 3)
        self.assertEqual(len(model.train_history.history["loss"]), 3)

    def test_mlp(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random(50)
        model = MLPEmbedding(2, time_window=5, random_state=0)
        latent = model.fit_transform(x, train_steps=3)
        self.assertEqual(latent.shape, (45, 2))

        model = MLPEmbedding(
            2, time_window=5, latent_regularizer=FNN(1.0), random_state=0
        )
        latent = model.fit_transform(x, train_steps=3)
        self.assertEqual(latent.shape, (45, 2))

    def test_causal_and_etd_constrained(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random(60)
        causal = CausalEmbedding(
            2, time_window=8, latent_regularizer=FNN(0.1), random_state=0
        )
        self.assertEqual(causal.fit_transform(x, train_steps=3).shape, (52, 2))

        etd_c = ETDConstrainedEmbedding(
            2, time_window=5, latent_regularizer=FNN(0.1), random_state=0
        )
        self.assertEqual(etd_c.fit_transform(x, train_steps=3).shape, (55, 2))

    def test_multivariate(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random((60, 2))
        model = MLPEmbedding(
            3, time_window=5, n_features=2, latent_regularizer=FNN(0.1), random_state=0
        )
        self.assertEqual(model.fit_transform(x, train_steps=3).shape, (55, 3))

    def test_baseline_models(self) -> None:
        t = np.linspace(0, 40, 200)
        sig = np.sin(t) + 0.3 * np.cos(2.7 * t)

        self.assertEqual(ETDEmbedding(3, time_window=10).fit_transform(sig).shape, (190, 3))
        self.assertEqual(TICAEmbedding(3, time_window=10, time_lag=5).fit_transform(sig).shape, (190, 3))
        self.assertEqual(TICAEmbedding(3, time_window=10, time_lag=0, random_state=0).fit_transform(sig).shape, (190, 3))
        self.assertEqual(ConstantLagEmbedding(3, time_window=5, lag_time=2).fit_transform(sig).shape, (190, 5))
        ami_out = AMIEmbedding(3, time_window=5, lag_cutoff=15).fit_transform(sig)
        self.assertEqual(ami_out.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
