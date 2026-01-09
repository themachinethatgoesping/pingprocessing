"""Request objects for echogram image building.

This module defines the data structures used to communicate between
the coordinate system and backends for efficient image generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class EchogramImageRequest:
    """Everything a backend needs to produce a downsampled echogram image.
    
    The coordinate system generates this request containing:
    - Output image dimensions (nx, ny)
    - Which ping maps to each output x column
    - Affine parameters for y→sample index conversion per ping
    
    The affine mapping is: sample_idx = round(a[p] + b[p] * y)
    where p is the ping index and y is the y-coordinate.
    
    Attributes:
        nx: Number of output x columns (image width).
        ny: Number of output y rows (image height).
        y_coordinates: The y-axis grid values, shape (ny,).
        ping_indexer: For each x column, which ping to use. Shape (nx,).
            Values are ping indices into backend data, or -1 for invalid.
        affine_a: Affine intercept per ping for y→sample mapping.
            Shape (n_pings,), NaN where undefined.
        affine_b: Affine slope per ping for y→sample mapping.
            Shape (n_pings,), NaN where undefined.
        max_sample_indices: Maximum valid sample index per ping.
            Shape (n_pings,). Used for bounds checking.
        fill_value: Value to use for invalid/missing data.
    """
    nx: int
    ny: int
    y_coordinates: np.ndarray
    ping_indexer: np.ndarray
    affine_a: np.ndarray
    affine_b: np.ndarray
    max_sample_indices: np.ndarray
    fill_value: float = np.nan
    
    def compute_sample_indices(self, ping_idx: int) -> np.ndarray:
        """Compute sample indices for a single ping.
        
        Args:
            ping_idx: The ping index (into affine arrays).
            
        Returns:
            Array of sample indices, shape (ny,). Invalid values are -1.
        """
        a = self.affine_a[ping_idx]
        b = self.affine_b[ping_idx]
        
        if np.isnan(a) or np.isnan(b):
            return np.full(self.ny, -1, dtype=np.int64)
        
        sample_idx = np.rint(a + b * self.y_coordinates).astype(np.int64)
        
        # Mark out-of-bounds as invalid
        max_idx = int(self.max_sample_indices[ping_idx])
        sample_idx[(sample_idx < 0) | (sample_idx >= max_idx)] = -1
        
        return sample_idx
    
    def compute_all_sample_indices(self) -> np.ndarray:
        """Compute sample indices for all pings at once (vectorized).
        
        Returns:
            Array of sample indices, shape (n_pings, ny). Invalid values are -1.
        """
        n_pings = len(self.affine_a)
        
        # Broadcast: a[:, None] + b[:, None] * y[None, :]
        a = self.affine_a[:, np.newaxis]  # (n_pings, 1)
        b = self.affine_b[:, np.newaxis]  # (n_pings, 1)
        y = self.y_coordinates[np.newaxis, :]  # (1, ny)
        
        # Where a or b is NaN, result will be NaN
        sample_idx_float = a + b * y  # (n_pings, ny)
        
        # Convert to int, NaN becomes some large negative
        sample_idx = np.rint(sample_idx_float).astype(np.int64)
        
        # Mark invalid: NaN in affine or out of bounds
        invalid_ping = np.isnan(self.affine_a) | np.isnan(self.affine_b)
        sample_idx[invalid_ping, :] = -1
        
        # Bounds check per ping
        max_idx = self.max_sample_indices[:, np.newaxis]  # (n_pings, 1)
        out_of_bounds = (sample_idx < 0) | (sample_idx >= max_idx)
        sample_idx[out_of_bounds] = -1
        
        return sample_idx
