"""Utilities for delay-embedding, dataset preprocessing, and plotting."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.io import arff
from scipy.linalg import hankel


###------------------------------------###
#
#       Loading and featurizing data
#
###------------------------------------###


def hankel_matrix(data: np.ndarray, q: int, p: int | None = None) -> np.ndarray:
    """Construct the Hankel (delay-embedding) matrix for univariate or multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        1D ``(T,)``, 2D ``(T, D)``, or 3D ``(N, T, D)`` time-series array.
    q : int
        Width of the delay window (number of lags).
    p : int or None, default=None
        Number of delay-embedded samples (rows); defaults to ``len(data) - q``.
    """
    data = np.asarray(data)
    if data.ndim == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])
    if data.ndim == 1:
        data = data[:, None]
    return _hankel_matrix(data, q, p)


def _hankel_matrix(data: np.ndarray, q: int, p: int | None = None) -> np.ndarray:
    """Calculate the Hankel matrix of a multivariate ``(T, D)`` time series."""
    if data.ndim == 1:
        data = data[:, None]

    if not p:
        p = len(data) - q

    all_hmats = []
    for row in data.T:
        first, last = row[-(p + q) : -p], row[-p - 1 :]
        all_hmats.append(hankel(first, last))
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))[:-1]


def resample_dataset(
    data: np.ndarray,
    n_samples: int | None = None,
    randomize: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows from a dataset without replacement, returning ``(indices, subset)``."""
    rng = np.random.default_rng(random_state)
    if not n_samples:
        n_samples = data.shape[0]
    if randomize:
        selection_indices = rng.choice(np.arange(data.shape[0]), n_samples, replace=False)
    else:
        selection_indices = np.arange(n_samples)
    return selection_indices, data[selection_indices]


def train_test(
    dataset: np.ndarray,
    sample_size: int,
    time_window: int,
    std: float = 1.0,
    split: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a time series into standardized train and test Hankel matrices."""
    n = len(dataset)
    n_split = int((split / (1 - split)) * sample_size)
    if n <= sample_size + n_split:
        raise ValueError("Not enough data to make complete split")

    hm_train = hankel_matrix(dataset, time_window, p=sample_size)
    hm_test = hankel_matrix(dataset[: (n_split + time_window)], time_window, p=sample_size)

    mn_train, std_train = np.mean(hm_train), np.std(hm_train)
    X_train, X_test = [(item - mn_train) / (std * std_train) for item in (hm_train, hm_test)]
    return X_train, X_test


def standardize_ts(a: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Standardize a ``(T, D)`` time series along its first dimension.

    Dimensions with zero variance are divided by 1 instead of 0.
    """
    a = np.asarray(a, dtype=float)
    stds = np.std(a, axis=0, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return (a - np.mean(a, axis=0, keepdims=True)) / (scale * stds)


def arff_to_data(path: str, fmt_spec: int = 1) -> tuple[np.ndarray, list]:
    """Load a UCR-format ARFF file into an ``(N, T, D)`` array and label list."""
    data, _ = arff.loadarff(path)
    y = [item[-1] for item in data]

    if fmt_spec == 0:
        data_rp = np.array([[tpt for tpt in sample][:-1] for sample in data], dtype=float)
    elif fmt_spec == 1:
        data_r = np.array([item[0] for item in data])
        data_rp = np.array(
            [[[float(timepoint) for timepoint in component] for component in trial] for trial in data_r]
        )
    else:
        warnings.warn("Format specification not implemented, falling back to fmt_spec=0", stacklevel=2)
        data_rp = np.array([[tpt for tpt in sample][:-1] for sample in data], dtype=float).T

    if data_rp.ndim == 2:
        data_rp = data_rp[:, None, :]

    data_rp = np.transpose(data_rp, (0, 2, 1))
    return data_rp, y


###------------------------------------###
#
#       Plotting and visualization
#
###------------------------------------###


def fixed_aspect_ratio(ratio: float, ax=None, log: bool = False) -> None:
    """Set a fixed aspect ratio on 2D matplotlib plots regardless of axis units."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    xvals, yvals = ax.get_xlim(), ax.get_ylim()
    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    if log:
        xrange = np.log(xvals[1]) - np.log(xvals[0])
        yrange = np.log(yvals[1]) - np.log(yvals[0])
    ax.set_aspect(ratio * (xrange / yrange), adjustable="box")


def plot_err(
    y: np.ndarray,
    errs: np.ndarray,
    color: tuple[float, float, float] = (0, 0, 0),
    x: np.ndarray | None = None,
    alpha: float = 0.4,
    linewidth: float = 1.0,
    **kwargs,
) -> None:
    """Plot a curve with shaded symmetric or asymmetric error bands."""
    import matplotlib.pyplot as plt

    if x is None or len(x) < 1:
        x = np.arange(len(y))

    errs = np.asarray(errs)
    if errs.ndim > 1 and errs.shape[1] == 2:
        err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = err_hi = errs

    plt.fill_between(x, y - err_lo, y + err_hi, color=lighter(color), alpha=alpha)
    plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)


def plot3dproj(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *args,
    ax=None,
    color: tuple[float, float, float] = (0, 0, 0),
    shadow_dist: float | tuple[float, float, float] = 1.0,
    color_proj: tuple[float, float, float] | None = None,
    elev_azim: tuple[float, float] = (39, -47),
    show_labels: bool = False,
    aspect_ratio: float | None = 1.0,
    **kwargs,
):
    """Create a 3D plot with projections onto the 2D coordinate planes."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
    if color_proj is None:
        color_proj = lighter(color, 0.6)

    if np.isscalar(shadow_dist):
        sdist_x = sdist_y = sdist_z = float(shadow_dist)
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist

    ax.plot(x, z, *args, zdir="y", zs=sdist_y * np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir="x", zs=sdist_x * np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir="z", zs=sdist_z * np.min(z), color=color_proj, **kwargs)
    ax.plot(x, y, z, *args, color=color, **kwargs)

    ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    if aspect_ratio:
        ax.set_box_aspect((1, 1, aspect_ratio))

    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    return ax


def lighter(clr: tuple[float, ...] | list[float], f: float = 1 / 3) -> list[float]:
    """Brighten an RGB color by fraction ``f``."""
    return [val + f * (1 - val) for val in clr]


def darker(clr: tuple[float, ...] | list[float], f: float = 1 / 3) -> list[float]:
    """Darken an RGB color by fraction ``f``."""
    return [val - f * val for val in clr]
