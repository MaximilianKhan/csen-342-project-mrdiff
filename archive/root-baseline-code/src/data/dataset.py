"""ETT Dataset implementation for mr-Diff.

Provides PyTorch Dataset for ETTh1 and ETTm1 time series data.
Uses per-window instance normalization (RevIN-style) as in the paper.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ETTDataset(Dataset):
    """PyTorch Dataset for ETT (Electricity Transformer Temperature) data.

    Supports both univariate and multivariate forecasting with
    sliding window approach. Uses per-window instance normalization
    where each sample is normalized by its own lookback mean and std.
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "ETTh1",
        lookback_length: int = 336,
        forecast_length: int = 168,
        split: str = "train",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        univariate: bool = False,
        eps: float = 1e-5,
    ):
        """Initialize the ETT dataset.

        Args:
            data_path: Path to the data directory containing ETT CSVs.
            dataset_name: Name of dataset ("ETTh1", "ETTm1", etc.).
            lookback_length: Number of historical time steps (H).
            forecast_length: Number of future time steps to predict (T).
            split: Data split ("train", "val", "test").
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            univariate: If True, use only the OT (last) column.
            eps: Small constant for numerical stability in normalization.
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length
        self.split = split
        self.univariate = univariate
        self.eps = eps

        # Load data (raw, unnormalized)
        self.data = self._load_data()

        # Split indices
        n_samples = len(self.data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        if split == "train":
            self.start_idx = 0
            self.end_idx = train_end
        elif split == "val":
            self.start_idx = train_end - lookback_length
            self.end_idx = val_end
        else:  # test
            self.start_idx = val_end - lookback_length
            self.end_idx = n_samples

        # Extract split data (raw, unnormalized)
        self.split_data = self.data[self.start_idx : self.end_idx]

        # Calculate number of windows
        self.num_windows = len(self.split_data) - lookback_length - forecast_length + 1

    def _load_data(self) -> np.ndarray:
        """Load the ETT dataset from CSV.

        Returns:
            Numpy array of shape [N, D] where N is time steps and D is features.
        """
        csv_path = self.data_path / f"{self.dataset_name}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. "
                f"Please run download_data.py first."
            )

        df = pd.read_csv(csv_path)

        # Drop the date column
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        data = df.values.astype(np.float32)

        if self.univariate:
            # Use only the last column (OT - Oil Temperature)
            data = data[:, -1:]

        return data

    def __len__(self) -> int:
        return max(0, self.num_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with per-window instance normalization.

        Each sample is normalized using the mean and std of its lookback window.
        This is the RevIN-style normalization used in the paper.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - "lookback": Normalized historical data [H, D]
                - "forecast": Normalized future data to predict [T, D]
                - "norm_mean": Mean used for normalization [D]
                - "norm_std": Std used for normalization [D]
        """
        start = idx
        lookback_end = start + self.lookback_length
        forecast_end = lookback_end + self.forecast_length

        # Get raw data
        lookback_raw = self.split_data[start:lookback_end]
        forecast_raw = self.split_data[lookback_end:forecast_end]

        # Convert to tensors
        lookback = torch.tensor(lookback_raw, dtype=torch.float32)
        forecast = torch.tensor(forecast_raw, dtype=torch.float32)

        # Compute per-window statistics from lookback (RevIN-style)
        # Mean and std computed along time dimension [H, D] -> [D]
        norm_mean = lookback.mean(dim=0, keepdim=True)  # [1, D]
        norm_std = lookback.std(dim=0, keepdim=True) + self.eps  # [1, D]

        # Normalize both lookback and forecast using lookback statistics
        lookback_norm = (lookback - norm_mean) / norm_std
        forecast_norm = (forecast - norm_mean) / norm_std

        return {
            "lookback": lookback_norm,
            "forecast": forecast_norm,
            "norm_mean": norm_mean.squeeze(0),  # [D]
            "norm_std": norm_std.squeeze(0),  # [D]
        }

    @property
    def num_features(self) -> int:
        """Number of features in the dataset."""
        return self.split_data.shape[1]


def create_dataloaders(
    data_path: str,
    dataset_name: str = "ETTh1",
    lookback_length: int = 336,
    forecast_length: int = 168,
    batch_size: int = 64,
    num_workers: int = 4,
    univariate: bool = False,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader, None]:
    """Create train, validation, and test dataloaders.

    Note: With per-window instance normalization, no global scaler is needed.
    The fourth return value is None for backward compatibility.

    Args:
        data_path: Path to the data directory.
        dataset_name: Name of dataset.
        lookback_length: Historical window size.
        forecast_length: Forecast horizon.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        univariate: Whether to use univariate data.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, None).
    """
    train_dataset = ETTDataset(
        data_path=data_path,
        dataset_name=dataset_name,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        univariate=univariate,
    )

    val_dataset = ETTDataset(
        data_path=data_path,
        dataset_name=dataset_name,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        univariate=univariate,
    )

    test_dataset = ETTDataset(
        data_path=data_path,
        dataset_name=dataset_name,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        univariate=univariate,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Compute global training statistics for evaluation metrics
    # Papers report metrics in globally-standardized space, not RevIN space
    from .preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data_tensor = torch.tensor(train_dataset.split_data, dtype=torch.float32)
    scaler.fit(train_data_tensor)

    return train_loader, val_loader, test_loader, scaler
