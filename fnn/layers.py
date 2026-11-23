"""Reusable PyTorch layers for time-series autoencoders."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(
    activation_func: nn.Module | Callable[[], nn.Module] | None,
) -> nn.Module:
    """Instantiate or clone an activation module (defaults to ``nn.ELU(alpha=1.0)``)."""
    if activation_func is None:
        return nn.ELU(alpha=1.0)
    if isinstance(activation_func, nn.Module):
        return copy.deepcopy(activation_func)
    if callable(activation_func):
        res = activation_func()
        if isinstance(res, nn.Module):
            return res
        raise TypeError("Callable activation_func must return an nn.Module instance.")
    raise TypeError(f"Expected nn.Module or callable, got {type(activation_func)}")


class GaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise during training only."""

    def __init__(self, std: float = 0.5) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class SequenceBatchNorm1d(nn.Module):
    """Apply BatchNorm1d across the channel dimension of ``(B, T, C)`` tensors."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class TimeMask(nn.Module):
    """Apply an exponential decay weight across the time dimension of ``(B, T, C)`` tensors."""

    def __init__(self, decay_rate: float = 0.0, reverse: bool = True) -> None:
        super().__init__()
        self.decay_rate = decay_rate
        self.reverse = reverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_time = x.shape[1]
        indices = torch.linspace(0.0, 1.0, n_time, dtype=x.dtype, device=x.device)
        mask = torch.exp(-self.decay_rate * indices)
        if self.reverse:
            mask = torch.flip(mask, dims=[0])
        return x * mask[None, :, None]


class Chomp1d(nn.Module):
    """Trim trailing ``chomp_size`` timesteps from a ``(B, T, C)`` sequence tensor."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, : -self.chomp_size, :]


class CausalConv1D(nn.Module):
    """Dilated causal 1D convolution (or transposed upsampling convolution) for ``(B, T, C)`` tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        strides: int = 1,
        dilation_rate: int = 1,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.is_transpose = transpose
        self.pad = (kernel_size - 1) * dilation_rate

        if transpose:
            self.conv: nn.Module = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                dilation=dilation_rate,
            )
            self.chomp = Chomp1d(self.pad)
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                dilation=dilation_rate,
            )
            self.chomp = Chomp1d(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        xt = x.transpose(1, 2)
        if self.is_transpose:
            out = self.conv(xt).transpose(1, 2)
            out = self.chomp(out)
            # Align output length to input_len * strides
            target_len = x.shape[1] * self.strides
            if out.shape[1] < target_len:
                out = F.pad(out, (0, 0, 0, target_len - out.shape[1]))
            elif out.shape[1] > target_len:
                out = out[:, :target_len, :]
            return out

        xt = F.pad(xt, (self.pad, 0))
        return self.conv(xt).transpose(1, 2)


class CausalBlock(nn.Module):
    """Residual dilated causal convolution block (Franceschi, Dieuleveut, Jaggi, NeurIPS 2019)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_rate: int = 1,
        transpose: bool = False,
        activation: nn.Module | Callable[[], nn.Module] | None = None,
        final: bool = False,
        strides: int = 1,
    ) -> None:
        super().__init__()
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)
        self.final_activation = nn.Identity() if final else make_activation(activation)

        self.cause1 = CausalConv1D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            strides=strides,
            transpose=transpose,
        )
        self.cause2 = CausalConv1D(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            strides=strides,
            transpose=transpose,
        )

        self.do_rescale = (in_channels != out_channels) or (strides > 1)
        if self.do_rescale:
            if transpose:
                self.rescale: nn.Module = nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size=1, stride=strides**2
                )
            else:
                self.rescale = nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=strides**2
                )
        else:
            self.rescale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.cause1(x))
        y = self.act2(self.cause2(y))
        if self.do_rescale:
            skip = self.rescale(x.transpose(1, 2)).transpose(1, 2)
            if skip.shape[1] < y.shape[1]:
                skip = F.pad(skip, (0, 0, 0, y.shape[1] - skip.shape[1]))
            elif skip.shape[1] > y.shape[1]:
                skip = skip[:, : y.shape[1], :]
            return self.final_activation(y + skip)
        return self.final_activation(y + x)


class CausalCNN(nn.Module):
    """Deep dilated causal convolutional network for ``(B, T, C)`` time series."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int | list[int] | tuple[int, ...],
        kernel_size: int = 3,
        strides: int = 1,
        transpose: bool = False,
        activation: nn.Module | Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]

        blocks: list[nn.Module] = []
        cur_in = in_channels
        for i, n_channel in enumerate(channels):
            blocks.append(
                CausalBlock(
                    cur_in,
                    n_channel,
                    kernel_size=kernel_size,
                    dilation_rate=2**i,
                    strides=strides,
                    transpose=transpose,
                    activation=activation,
                    final=False,
                )
            )
            cur_in = n_channel

        blocks.append(
            CausalBlock(
                cur_in,
                out_channels,
                kernel_size=kernel_size,
                dilation_rate=1,
                strides=1,
                transpose=transpose,
                activation=activation,
                final=True,
            )
        )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
