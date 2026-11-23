"""PyTorch autoencoder architectures for time-series embedding."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .layers import CausalConv1D, GaussianNoise, SequenceBatchNorm1d, make_activation


class _PredictMixin:
    """Convenience inference helper accepting NumPy arrays or PyTorch tensors."""

    @torch.inference_mode()
    def predict(
        self,
        x: np.ndarray | torch.Tensor,
        batch_size: int | None = None,
    ) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            param = next(self.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
            dtype = param.dtype if param is not None else torch.float32
            tensor = torch.as_tensor(x, dtype=dtype)

            if batch_size is None:
                return self(tensor.to(device)).cpu().numpy()

            outputs = []
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start : start + batch_size].to(device)
                outputs.append(self(batch).cpu().numpy())
            return np.concatenate(outputs, axis=0)
        finally:
            if was_training:
                self.train()


class _AutoencoderBase(nn.Module):
    """Base PyTorch autoencoder holding an encoder, decoder, and optional latent regularizer."""

    encoder: nn.Module
    decoder: nn.Module
    latent_regularizer: Callable[[torch.Tensor], torch.Tensor] | None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_with_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon, code

    def regularization_loss(self, code: torch.Tensor) -> torch.Tensor:
        if self.latent_regularizer is None:
            return code.new_zeros(())
        return self.latent_regularizer(code)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _MLPEncoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        time_window: int,
        n_features: int,
        n_latent: int,
        network_shape: Sequence[int],
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(), GaussianNoise(noise_std)]
        in_dim = time_window * n_features
        for hidden_size in network_shape:
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                make_activation(activation_func),
            ])
            in_dim = hidden_size
        layers.extend([
            nn.Linear(in_dim, n_latent),
            nn.BatchNorm1d(n_latent),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MLPDecoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        time_window: int,
        n_features: int,
        n_latent: int,
        network_shape: Sequence[int],
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.time_window = time_window
        self.n_features = n_features
        layers: list[nn.Module] = [nn.Flatten(), GaussianNoise(noise_std)]
        in_dim = n_latent
        for hidden_size in reversed(network_shape):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                make_activation(activation_func),
            ])
            in_dim = hidden_size
        layers.extend([
            nn.Linear(in_dim, time_window * n_features),
            nn.BatchNorm1d(time_window * n_features),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1, self.time_window, self.n_features)


class MLPAutoencoder(_AutoencoderBase):
    """Fully-connected time-delay autoencoder for time series."""

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        network_shape: Sequence[int] = (10, 10),
        latent_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        rnn_opts: dict[str, Any] | None = None,
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        random_state: int | None = None,
        noise_std: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = rnn_opts, kwargs
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.latent_regularizer = latent_regularizer

        if random_state is not None:
            torch.manual_seed(random_state)

        self.encoder = _MLPEncoder(
            time_window,
            n_features,
            n_latent,
            network_shape=network_shape,
            activation_func=activation_func,
            noise_std=noise_std,
        )
        self.decoder = _MLPDecoder(
            time_window,
            n_features,
            n_latent,
            network_shape=network_shape,
            activation_func=activation_func,
            noise_std=noise_std,
        )


class _LSTMEncoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        time_window: int,
        n_features: int,
        n_latent: int,
        network_shape: Sequence[int] = (),
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.time_window = time_window
        self.noise = GaussianNoise(noise_std)
        self.hidden_lstms = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()

        in_dim = n_features
        for hidden_size in network_shape:
            self.hidden_lstms.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
            self.hidden_bns.append(SequenceBatchNorm1d(hidden_size))
            self.hidden_acts.append(make_activation(activation_func))
            in_dim = hidden_size

        self.final_lstm = nn.LSTM(in_dim, n_latent, batch_first=True)
        self.final_bn = nn.BatchNorm1d(n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.noise(x)
        for lstm, bn, act in zip(self.hidden_lstms, self.hidden_bns, self.hidden_acts):
            h, _ = lstm(h)
            h = act(bn(h))
        _, (h_n, _) = self.final_lstm(h)
        return self.final_bn(h_n[-1])


class _LSTMDecoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        time_window: int,
        n_features: int,
        n_latent: int,
        network_shape: Sequence[int] = (),
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.time_window = time_window
        self.noise = GaussianNoise(noise_std)
        self.initial_lstm = nn.LSTM(n_latent, n_latent, batch_first=True)
        self.hidden_lstms = nn.ModuleList()
        in_dim = n_latent
        for hidden_size in reversed(network_shape):
            self.hidden_lstms.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
            in_dim = hidden_size

        self.bn = SequenceBatchNorm1d(in_dim)
        self.activation = make_activation(activation_func)
        self.out_proj = nn.Linear(in_dim, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq = self.noise(z).unsqueeze(1).expand(-1, self.time_window, -1)
        h, _ = self.initial_lstm(seq)
        for lstm in self.hidden_lstms:
            h, _ = lstm(h)
        h = self.activation(self.bn(h))
        return self.out_proj(h)


class LSTMAutoencoder(_AutoencoderBase):
    """LSTM autoencoder for time series."""

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        network_shape: Sequence[int] = (),
        latent_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        rnn_opts: dict[str, Any] | None = None,
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        random_state: int | None = None,
        noise_std: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = rnn_opts, kwargs
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.latent_regularizer = latent_regularizer

        if random_state is not None:
            torch.manual_seed(random_state)

        self.encoder = _LSTMEncoder(
            time_window,
            n_features,
            n_latent,
            network_shape=network_shape,
            activation_func=activation_func,
            noise_std=noise_std,
        )
        self.decoder = _LSTMDecoder(
            time_window,
            n_features,
            n_latent,
            network_shape=network_shape,
            activation_func=activation_func,
            noise_std=noise_std,
        )


class _CausalAEEncoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        n_features: int,
        n_latent: int,
        network_shape: Sequence[int],
        kernel_size: int,
        dilation_scale: int,
        strides: int,
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [GaussianNoise(noise_std)]
        in_ch = n_features
        for i, hidden_size in enumerate(network_shape):
            layers.extend([
                CausalConv1D(
                    in_ch,
                    hidden_size,
                    kernel_size=kernel_size,
                    strides=strides,
                    dilation_rate=dilation_scale**i,
                ),
                SequenceBatchNorm1d(hidden_size),
                make_activation(activation_func),
            ])
            in_ch = hidden_size
        self.conv_net = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_ch, n_latent, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_net(x)
        _, (h_n, _) = self.lstm(h)
        return h_n[-1]


class _CausalAEDecoder(_PredictMixin, nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_features: int,
        time_window_downsampled: int,
        network_shape: Sequence[int],
        kernel_size: int,
        dilation_scale: int,
        strides: int,
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.time_window_downsampled = time_window_downsampled
        self.noise = GaussianNoise(noise_std)
        self.lstm = nn.LSTM(n_latent, network_shape[-1], batch_first=True)
        self.lstm_act = make_activation(activation_func)

        deconv_layers: list[nn.Module] = []
        in_ch = network_shape[-1]
        out_channels_list = list(reversed(network_shape))[1:] + [n_features]
        for i, hidden_unit in enumerate(out_channels_list):
            upsamp_factor = dilation_scale ** (len(network_shape) - i - 1)
            deconv_layers.append(
                CausalConv1D(
                    in_ch,
                    hidden_unit,
                    kernel_size=kernel_size,
                    strides=strides,
                    dilation_rate=upsamp_factor,
                    transpose=True,
                )
            )
            deconv_layers.append(SequenceBatchNorm1d(hidden_unit))
            if i < len(out_channels_list) - 1:
                deconv_layers.append(make_activation(activation_func))
            in_ch = hidden_unit
        self.deconv_net = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        seq = self.noise(z).unsqueeze(1).expand(-1, self.time_window_downsampled, -1)
        h, _ = self.lstm(seq)
        h = self.lstm_act(h)
        return self.deconv_net(h)


class CausalAE(_AutoencoderBase):
    """Dilated causal convolutional + LSTM autoencoder for time series."""

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        network_shape: Sequence[int] = (10, 10),
        latent_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        kernel_size: int = 3,
        dilation_scale: int = 2,
        strides: int = 1,
        rnn_opts: dict[str, Any] | None = None,
        activation_func: nn.Module | Callable[[], nn.Module] | None = None,
        random_state: int | None = None,
        noise_std: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = rnn_opts, kwargs
        self.n_latent = n_latent
        self.n_features = n_features
        self.latent_regularizer = latent_regularizer

        time_window_downsampled = math.ceil(time_window / (strides ** len(network_shape)))
        self.time_window = time_window_downsampled * (strides ** len(network_shape))

        if random_state is not None:
            torch.manual_seed(random_state)

        self.encoder = _CausalAEEncoder(
            n_features,
            n_latent,
            network_shape=network_shape,
            kernel_size=kernel_size,
            dilation_scale=dilation_scale,
            strides=strides,
            activation_func=activation_func,
            noise_std=noise_std,
        )
        self.decoder = _CausalAEDecoder(
            n_latent,
            n_features,
            time_window_downsampled=time_window_downsampled,
            network_shape=network_shape,
            kernel_size=kernel_size,
            dilation_scale=dilation_scale,
            strides=strides,
            activation_func=activation_func,
            noise_std=noise_std,
        )


class _LinearEncoder(_PredictMixin, nn.Module):
    def __init__(self, time_window: int, n_features: int, n_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(time_window * n_features, n_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LinearDecoder(_PredictMixin, nn.Module):
    def __init__(self, time_window: int, n_features: int, n_latent: int) -> None:
        super().__init__()
        self.time_window = time_window
        self.n_features = n_features
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_latent, time_window * n_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1, self.time_window, self.n_features)


class ETDConstrained(_AutoencoderBase):
    """Linear autoencoder implementing a regularized eigen-time-delay embedding."""

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        latent_regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        rnn_opts: dict[str, Any] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = rnn_opts, kwargs
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.latent_regularizer = latent_regularizer

        if random_state is not None:
            torch.manual_seed(random_state)

        self.encoder = _LinearEncoder(time_window, n_features, n_latent)
        self.decoder = _LinearDecoder(time_window, n_features, n_latent)
