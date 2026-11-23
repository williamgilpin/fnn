"""Scikit-learn-style time-series embedding models backed by PyTorch."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import argrelextrema, savgol_filter
from sklearn.decomposition import PCA, FastICA, KernelPCA, SparsePCA
from sklearn.metrics import mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

from .networks import CausalAE, ETDConstrained, LSTMAutoencoder, MLPAutoencoder, _AutoencoderBase
from .regularizers import FNN
from .tica import tICA
from .utils import hankel_matrix, resample_dataset, standardize_ts


class TrainHistory(dict):
    """Dictionary of per-epoch training metrics (also accessible via ``.history``)."""

    @property
    def history(self) -> TrainHistory:
        return self


class TimeSeriesEmbedding:
    """Base class for time series embedding models.

    Parameters
    ----------
    n_latent : int
        Embedding dimension.
    time_window : int, default=10
        Number of delay-embedding timesteps per input window.
    n_features : int, default=1
        Number of channels in the input time series.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int = 10,
        n_features: int = 1,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None, **kwargs: Any) -> TimeSeriesEmbedding:
        raise NotImplementedError("Derived class must implement fit().")

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        raise NotImplementedError("Derived class must implement transform().")

    def fit_transform(self, X: np.ndarray, y: Any = None, **kwargs: Any) -> np.ndarray:
        """Fit the model with a time series ``X`` and return its embedding."""
        self.fit(X, **kwargs)
        return self.transform(X)


class ETDEmbedding(TimeSeriesEmbedding):
    """Embed time series using PCA / Eigen-time-delay (Broomhead-King) coordinates."""

    def __init__(
        self,
        *args: Any,
        sparse: bool = False,
        kernel: str | Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if kernel:
            self.model: Any = KernelPCA(
                n_components=self.n_latent,
                kernel=kernel,
                random_state=self.random_state,
                copy_X=False,
            )
        elif sparse:
            self.model = SparsePCA(
                n_components=self.n_latent,
                random_state=self.random_state,
            )
        else:
            self.model = PCA(
                n_components=self.n_latent,
                random_state=self.random_state,
            )

    def fit(self, X: np.ndarray, y: Any = None, subsample: int | None = None) -> ETDEmbedding:
        _ = y
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)
        if subsample:
            self.train_indices, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        _ = y
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        return self.model.transform(X_test)


class ConstantLagEmbedding(TimeSeriesEmbedding):
    """Embed a time series using a fixed delay lag (Takens, 1981)."""

    def __init__(
        self,
        *args: Any,
        lag_time: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lag_time = lag_time

    def fit(self, X: np.ndarray, y: Any = None, lag_cutoff: int | None = None) -> ConstantLagEmbedding:
        _ = X, y, lag_cutoff
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        _ = y
        tau = int(self.time_window * self.lag_time)
        X_test = hankel_matrix(standardize_ts(X), q=tau, p=len(X) - tau)
        X_test = X_test[:, :: int(self.lag_time), :]
        return np.squeeze(X_test)


class AMIEmbedding(ConstantLagEmbedding):
    """Embed a time series using delay lag selected by average mutual information (Fraser & Swinney, 1986)."""

    def __init__(
        self,
        *args: Any,
        lag_cutoff: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lag_cutoff = lag_cutoff

    @staticmethod
    def _compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int | None = None) -> float:
        n_bins = bins if bins else min(len(x), 64)
        contingency = np.histogram2d(x, y, n_bins)[0]
        return float(mutual_info_score(None, None, contingency=contingency))

    @classmethod
    def _mutual_information_lagged(
        cls,
        data: np.ndarray,
        max_time: int,
        bins: int | None = None,
    ) -> np.ndarray:
        series = np.asarray(data).squeeze()
        all_mi = []
        for tau in range(1, max_time):
            unlagged = series[:-tau]
            lagged = series[tau:]
            all_mi.append(cls._compute_mutual_information(unlagged, lagged, bins))
        return np.array(all_mi)

    @staticmethod
    def _find_minima(x: np.ndarray, smoothing_radius: int | None = None):
        if smoothing_radius:
            if (smoothing_radius % 2) == 0:
                smoothing_radius += 1
            x = savgol_filter(x, smoothing_radius, 3)
        return argrelextrema(x, np.less)

    def fit(
        self,
        X: np.ndarray,
        y: Any = None,
        verbose: bool = False,
        bins: int | None = None,
        timescale: int | None = None,
    ) -> AMIEmbedding:
        _ = y
        Xs = standardize_ts(X)
        cutoff = self.lag_cutoff if self.lag_cutoff else int(np.floor(len(Xs) / 2))
        lagged_mi_vals = self._mutual_information_lagged(Xs, cutoff, bins)
        if verbose:
            import matplotlib.pyplot as plt

            plt.plot(lagged_mi_vals)

        lag_times = self._find_minima(lagged_mi_vals, timescale)[0]
        self.lag_time = int(lag_times[0]) + 1 if len(lag_times) > 0 else 1
        return self


class TICAEmbedding(TimeSeriesEmbedding):
    """Embed time series using time-lagged independent component analysis (tICA) or FastICA."""

    def __init__(
        self,
        *args: Any,
        time_lag: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.time_lag = time_lag
        if time_lag > 0:
            self.model: Any = tICA(n_components=self.n_latent, lag_time=time_lag)
        elif time_lag == 0:
            self.model = FastICA(n_components=self.n_latent, random_state=self.random_state)
        else:
            raise ValueError("Time delay parameter must be greater than or equal to zero.")

    def fit(self, X: np.ndarray, y: Any = None, subsample: int | None = None) -> TICAEmbedding:
        _ = y
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)
        if subsample:
            self.train_indices, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
        flat = np.reshape(X_train, (X_train.shape[0], -1))
        if self.time_lag > 0:
            self.model.fit([flat])
        else:
            self.model.fit(flat)
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        _ = y
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        flat = np.reshape(X_test, (X_test.shape[0], -1))
        if self.time_lag > 0:
            return self.model.transform([flat])[0]
        return self.model.transform(flat)


class NeuralNetworkEmbedding(TimeSeriesEmbedding):
    """Base class for PyTorch autoencoder time-series embeddings."""

    model: _AutoencoderBase

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.train_history: TrainHistory = TrainHistory()

    @staticmethod
    def _resolve_loss(loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        if callable(loss):
            return loss
        losses = {
            "mse": F.mse_loss,
            "mean_squared_error": F.mse_loss,
            "mae": F.l1_loss,
            "l1": F.l1_loss,
            "huber": F.huber_loss,
        }
        if loss not in losses:
            raise ValueError(f"Unsupported loss {loss!r}. Choose from {sorted(losses)} or pass a callable.")
        return losses[loss]

    @staticmethod
    def _resolve_optimizer(name: str, params, learning_rate: float) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "nadam": torch.optim.NAdam,
            "sgd": torch.optim.SGD,
        }
        key = name.lower()
        if key not in optimizers:
            raise ValueError(f"Unsupported optimizer {name!r}. Choose from {sorted(optimizers)}.")
        return optimizers[key](params, lr=learning_rate)

    def fit(
        self,
        X: np.ndarray,
        y: Any = None,
        subsample: int | None = None,
        tau: int = 0,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        train_steps: int = 200,
        loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "mse",
        verbose: int = 0,
        optimizer: str = "adam",
        early_stopping: bool = False,
        device: str | torch.device | None = None,
    ) -> NeuralNetworkEmbedding:
        """Fit the autoencoder on a time series ``X``.

        Parameters
        ----------
        X : array-like of shape ``(n_timepoints,)`` or ``(n_timepoints, n_features)``
            Input time series.
        subsample : int or None, default=None
            Optional number of delay windows to subsample randomly without replacement.
        tau : int, default=0
            Forecast offset (number of timesteps between input and target windows).
        learning_rate : float, default=1e-3
            Optimizer learning rate.
        batch_size : int, default=100
            Mini-batch size.
        train_steps : int, default=200
            Number of training epochs.
        loss : str or callable, default='mse'
            Reconstruction loss function.
        verbose : int, default=0
            If > 0, periodically prints training progress.
        optimizer : str, default='adam'
            Optimizer name (``'adam'``, ``'adamw'``, ``'nadam'``, or ``'sgd'``).
        early_stopping : bool, default=False
            If True, stops training after 3 consecutive epochs without loss improvement.
        device : str, torch.device, or None, default=None
            Target device; defaults to CUDA if available, else CPU.
        """
        _ = y
        Xs = standardize_ts(X)
        X0 = hankel_matrix(Xs, self.time_window + tau)
        X_train = X0[:, : self.time_window]
        Y_train = X0[:, -self.time_window :]

        if subsample:
            self.train_indices, _ = resample_dataset(
                X_train, subsample, random_state=self.random_state
            )
            X_train = X_train[self.train_indices]
            Y_train = Y_train[self.train_indices]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        target_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(target_device)

        loss_fn = self._resolve_loss(loss)
        opt = self._resolve_optimizer(optimizer, self.model.parameters(), learning_rate)

        dataset = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(Y_train, dtype=torch.float32),
        )
        drop_last = len(dataset) > batch_size and (len(dataset) % batch_size == 1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        history = TrainHistory(loss=[], mse_loss=[], reg_loss=[])
        best_loss = float("inf")
        stale_epochs = 0

        self.model.train()
        for epoch in range(train_steps):
            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_reg = 0.0
            n_batches = 0

            for bx, by in loader:
                bx = bx.to(target_device)
                by = by.to(target_device)
                opt.zero_grad(set_to_none=True)

                recon, code = self.model.forward_with_latent(bx)
                recon_loss = loss_fn(recon, by)
                reg_loss = self.model.regularization_loss(code)
                total_loss = recon_loss + reg_loss

                total_loss.backward()
                opt.step()

                epoch_total += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_reg += reg_loss.item()
                n_batches += 1

            mean_total = epoch_total / max(1, n_batches)
            mean_recon = epoch_recon / max(1, n_batches)
            mean_reg = epoch_reg / max(1, n_batches)
            history["loss"].append(mean_total)
            history["mse_loss"].append(mean_recon)
            history["reg_loss"].append(mean_reg)

            if verbose and ((epoch + 1) % max(1, train_steps // 10) == 0 or epoch == 0):
                print(
                    f"Epoch {epoch + 1}/{train_steps} - "
                    f"loss: {mean_total:.6f} - recon: {mean_recon:.6f} - reg: {mean_reg:.6f}"
                )

            if early_stopping:
                if mean_total < best_loss - 1e-7:
                    best_loss = mean_total
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= 3:
                        break

        self.model.eval()
        self.train_history = history
        return self

    def transform(self, X: np.ndarray, y: Any = None, batch_size: int | None = None) -> np.ndarray:
        _ = y
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        return self.model.encoder.predict(X_test, batch_size=batch_size)


class ETDConstrainedEmbedding(NeuralNetworkEmbedding):
    """Linear autoencoder embedding with optional latent regularization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        net_kwargs = {k: v for k, v in kwargs.items() if k != "time_window"}
        self.model = ETDConstrained(self.n_latent, self.time_window, **net_kwargs)


class MLPEmbedding(NeuralNetworkEmbedding):
    """Feedforward time-delay neural network (MLP) embedding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        net_kwargs = {k: v for k, v in kwargs.items() if k != "time_window"}
        self.model = MLPAutoencoder(self.n_latent, self.time_window, **net_kwargs)


class LSTMEmbedding(NeuralNetworkEmbedding):
    """LSTM autoencoder time-series embedding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        net_kwargs = {k: v for k, v in kwargs.items() if k != "time_window"}
        self.model = LSTMAutoencoder(self.n_latent, self.time_window, **net_kwargs)


class CausalEmbedding(NeuralNetworkEmbedding):
    """Dilated causal convolutional + LSTM time-series embedding."""

    def __init__(
        self,
        *args: Any,
        strides: int = 1,
        network_shape: Sequence[int] = (10, 10),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        net_kwargs = {k: v for k, v in kwargs.items() if k != "time_window"}
        time_window_downsampled = math.ceil(self.time_window / (strides ** len(network_shape)))
        time_window_upsampled = time_window_downsampled * (strides ** len(network_shape))
        if time_window_upsampled != self.time_window:
            self.time_window = time_window_upsampled

        self.model = CausalAE(
            self.n_latent,
            self.time_window,
            strides=strides,
            network_shape=network_shape,
            **net_kwargs,
        )
