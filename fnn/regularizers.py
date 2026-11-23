"""PyTorch activity regularizers for false-nearest-neighbor time-series embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class FNN(nn.Module):
    """Activity regularizer that penalizes false nearest neighbors in latent space.

    Parameters
    ----------
    strength : float, default=1.0
        Relative weight of the regularizer.
    k : int, default=1
        Number of nearest neighbors used to evaluate neighborhood unfolding.
    rtol : float, default=20.0
        Relative distance threshold (Kennel et al., 1992, Eq. 4).
    atol : float, default=2.0
        Absolute attractor-radius threshold (Kennel et al., 1992, Eq. 5).
    """

    def __init__(
        self,
        strength: float = 1.0,
        k: int = 1,
        rtol: float = 20.0,
        atol: float = 2.0,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.k = k
        self.rtol = rtol
        self.atol = atol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * loss_false(x, k=self.k, rtol=self.rtol, atol=self.atol)


class DeCov(nn.Module):
    """Activity regularizer that encourages orthogonal latent coordinates (Cogswell et al., ICLR 2016).

    Parameters
    ----------
    strength : float, default=1.0
        Relative weight of the regularizer.
    whiten : bool, default=False
        Whether to standardize activations feature-wise before computing covariance.
    """

    def __init__(self, strength: float = 1.0, whiten: bool = False) -> None:
        super().__init__()
        self.strength = strength
        self.whiten = whiten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * loss_cov(x, whiten=self.whiten)


class L1Reg(nn.Module):
    """L1 activity regularizer for latent representations."""

    def __init__(self, strength: float = 0.01) -> None:
        super().__init__()
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * x.abs().mean()


class L2Reg(nn.Module):
    """L2 activity regularizer for latent representations."""

    def __init__(self, strength: float = 0.01) -> None:
        super().__init__()
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * x.pow(2).mean()


def loss_false(
    code_batch: torch.Tensor,
    k: int = 1,
    rtol: float = 20.0,
    atol: float = 2.0,
) -> torch.Tensor:
    """False-nearest-neighbor activity regularizer (Kennel, Brown, and Abarbanel, Phys. Rev. A 1992).

    Parameters
    ----------
    code_batch : torch.Tensor
        Encoded latent activations with shape ``(batch_size, n_latent)``.
    k : int, default=1
        Number of nearest neighbors used to compute neighborhoods.
    rtol : float, default=20.0
        Relative distance tolerance (Kennel et al., Eq. 4).
    atol : float, default=2.0
        Absolute attractor-radius tolerance (Kennel et al., Eq. 5).

    Returns
    -------
    torch.Tensor
        Scalar false-nearest-neighbor regularization penalty.
    """
    if code_batch.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor of shape (batch_size, n_latent), got shape {tuple(code_batch.shape)}"
        )

    n_batch, n_latent = code_batch.shape
    if n_batch < 2 or n_latent < 2:
        return code_batch.new_zeros(())

    n_neighbors = min(max(1, int(k)), n_batch - 1)
    compute_dtype = torch.float64 if code_batch.dtype == torch.float64 else torch.float32

    # Neighbor indicator weights are piecewise constant (zero gradient), so
    # compute them outside the autograd graph.
    with torch.no_grad():
        z = code_batch.detach().to(compute_dtype)

        # Cumulative squared pairwise distances across dimensions 1..n_latent
        diff_sq = (z[:, None, :] - z[None, :, :]).pow(2)  # (B, B, L)
        all_dists = torch.cumsum(diff_sq, dim=-1).permute(2, 0, 1)  # (L, B, B)
        all_dists = all_dists.clamp(min=1e-14)

        # Attractor scale R_A across dimensions 1..n_latent
        dims = torch.arange(1, n_latent + 1, dtype=compute_dtype, device=z.device)
        all_ra = torch.sqrt(torch.cumsum(z.var(dim=0, unbiased=False), dim=0) / dims)

        # Nearest neighbors (k + 1 including self at index 0)
        _, inds = torch.topk(all_dists, n_neighbors + 1, dim=-1, largest=False)

        neighbor_dists_d = torch.gather(all_dists, dim=-1, index=inds)
        neighbor_new_dists = torch.gather(all_dists[1:], dim=-1, index=inds[:-1])

        # Kennel et al. Eq. 4 (all_dists holds squared distances)
        scaled_dist = torch.sqrt(
            (neighbor_new_dists - neighbor_dists_d[:-1]).clamp(min=0.0)
            / neighbor_dists_d[:-1]
        )

        # Kennel conditions #1 and #2
        is_false_change = scaled_dist > rtol
        is_large_jump = neighbor_new_dists > atol * all_ra[:-1, None, None]
        is_false_neighbor = is_false_change | is_large_jump

        # Exclude self-neighbor at index 0
        false_fraction = (
            is_false_neighbor[..., 1 : n_neighbors + 1]
            .to(compute_dtype)
            .mean(dim=(1, 2))
        )

        # Leading zero ensures the first latent dimension is never shrunk to zero
        reg_weights = torch.cat([z.new_zeros(1), 1.0 - false_fraction])

    # Batch-averaged RMS activity per latent unit (eps avoids NaN grad at zero)
    activations_rms = torch.sqrt(code_batch.to(compute_dtype).pow(2).mean(dim=0) + 1e-24)
    loss = torch.sum(reg_weights * activations_rms)
    return loss.to(code_batch.dtype)


def loss_cov(a: torch.Tensor, whiten: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Covariance loss used to orthogonalize latent activations (Cogswell et al., ICLR 2016).

    Parameters
    ----------
    a : torch.Tensor
        Layer activations with shape ``(batch_size, n_features)``.
    whiten : bool, default=False
        Whether to standardize activations feature-wise before computing covariance.
    eps : float, default=1e-8
        Lower bound on feature standard deviation when ``whiten=True``.
    """
    aw = a - a.mean(dim=0, keepdim=True)
    if whiten:
        a_std = a.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        aw = aw / a_std

    cov = (aw.T @ aw) / a.shape[0]
    diag = torch.diagonal(cov)
    return 0.5 * (cov.pow(2).sum() - diag.pow(2).sum())
