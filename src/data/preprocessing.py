"""Preprocessing utilities for mr-Diff.

Includes trend extraction and instance normalization.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendExtraction(nn.Module):
    """Multi-resolution trend extraction using average pooling.

    Extracts trend components at different resolutions using
    increasing kernel sizes (τs) at each stage.
    """

    def __init__(self, kernel_sizes: List[int]):
        """Initialize trend extraction.

        Args:
            kernel_sizes: List of kernel sizes [τ1, τ2, ..., τS-1].
                         Each τs should be odd for symmetric padding.
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_stages = len(kernel_sizes) + 1  # S stages (0 to S-1)

        # Validate kernel sizes are odd
        for i, k in enumerate(kernel_sizes):
            if k % 2 == 0:
                raise ValueError(f"Kernel size at stage {i} must be odd, got {k}")

    def extract_trend(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Extract trend using average pooling.

        Args:
            x: Input tensor [B, T, D] or [B, D, T].
            kernel_size: Size of the averaging kernel.

        Returns:
            Trend component with same shape as input.
        """
        # Assume input is [B, T, D], convert to [B, D, T] for conv
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim

        # x: [B, T, D] -> [B, D, T]
        x_transposed = x.transpose(1, 2)

        # Pad to maintain sequence length (symmetric padding)
        pad_size = kernel_size // 2
        x_padded = F.pad(x_transposed, (pad_size, pad_size), mode="replicate")

        # Apply average pooling
        trend = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)

        # Convert back to [B, T, D]
        return trend.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Decompose signal into multi-resolution residual components.

        Produces a residual decomposition where x = sum(components):
            components[0] = x - trend_1       (finest residual)
            components[1] = trend_1 - trend_2
            ...
            components[S-1] = coarsest trend

        Args:
            x: Input tensor [B, T, D].

        Returns:
            List of S tensors whose sum reconstructs the original signal.
        """
        components = []
        current = x

        for kernel_size in self.kernel_sizes:
            trend = self.extract_trend(current, kernel_size)
            residual = current - trend
            components.append(residual)
            current = trend

        # Add the final trend (coarsest)
        components.append(current)

        return components

    def get_trends_and_residuals(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get both trends and residuals for visualization.

        Args:
            x: Input tensor [B, T, D].

        Returns:
            Tuple of (trends, residuals) lists.
        """
        trends = []
        residuals = []
        current = x

        for kernel_size in self.kernel_sizes:
            trend = self.extract_trend(current, kernel_size)
            residual = current - trend
            trends.append(trend)
            residuals.append(residual)
            current = trend

        # Final trend
        trends.append(current)
        residuals.append(torch.zeros_like(current))  # No residual at coarsest level

        return trends, residuals


class InstanceNormalization(nn.Module):
    """Instance normalization for time series.

    Normalizes each sample independently by subtracting mean
    and dividing by standard deviation (RevIN-style).
    """

    def __init__(self, eps: float = 1e-5, affine: bool = False):
        """Initialize instance normalization.

        Args:
            eps: Small constant for numerical stability.
            affine: Whether to learn affine parameters.
        """
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            # Learnable parameters (initialized after first forward)
            self.gamma = None
            self.beta = None

    def forward(
        self, x: torch.Tensor, return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the input.

        Args:
            x: Input tensor [B, T, D].
            return_stats: Whether to return normalization statistics.

        Returns:
            Normalized tensor, and optionally (mean, std).
        """
        # Compute statistics over time dimension
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        std = x.std(dim=1, keepdim=True) + self.eps  # [B, 1, D]

        # Normalize
        x_norm = (x - mean) / std

        # Apply affine transformation if enabled
        if self.affine:
            if self.gamma is None:
                self.gamma = nn.Parameter(torch.ones(1, 1, x.size(-1), device=x.device))
                self.beta = nn.Parameter(torch.zeros(1, 1, x.size(-1), device=x.device))
            x_norm = x_norm * self.gamma + self.beta

        if return_stats:
            return x_norm, mean, std
        return x_norm

    def denormalize(
        self, x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize the input using stored statistics.

        Args:
            x_norm: Normalized tensor [B, T, D].
            mean: Mean used for normalization [B, 1, D].
            std: Std used for normalization [B, 1, D].

        Returns:
            Denormalized tensor.
        """
        if self.affine and self.gamma is not None:
            x_norm = (x_norm - self.beta) / self.gamma

        return x_norm * std + mean


class StandardScaler:
    """Standard scaler for dataset-level normalization.

    Fits on training data and transforms all splits.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.eps = 1e-5

    def fit(self, data: torch.Tensor) -> "StandardScaler":
        """Fit the scaler on training data.

        Args:
            data: Training data [N, T, D] or [N, D].

        Returns:
            Self for chaining.
        """
        # Flatten all but the last dimension
        if data.dim() > 2:
            data = data.reshape(-1, data.size(-1))

        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + self.eps

        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted statistics.

        Args:
            data: Data to transform.

        Returns:
            Transformed data.
        """
        if self.mean is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform normalized data.

        Args:
            data: Normalized data.

        Returns:
            Original scale data.
        """
        if self.mean is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        # Move mean/std to same device as data
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step.

        Args:
            data: Training data.

        Returns:
            Transformed data.
        """
        return self.fit(data).transform(data)
