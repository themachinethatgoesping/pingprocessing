"""Storage axis mode definitions for echogram backends.

This module defines how echogram data is stored (what coordinate system),
which affects how backends transform data for display.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np


class XAxisType(Enum):
    """X-axis storage types."""
    PING_INDEX = "ping_index"  # Integer ping indices (0, 1, 2, ...)
    PING_TIME = "ping_time"    # Unix timestamps (seconds)


class YAxisType(Enum):
    """Y-axis storage types."""
    SAMPLE_INDEX = "sample_index"  # Integer sample indices (0, 1, 2, ...)
    SAMPLE_NR = "sample_nr"        # Sample numbers (may have offset)
    DEPTH = "depth"                # Depth in meters
    RANGE = "range"                # Range in meters


class ResolutionStrategy(Enum):
    """Strategy for resolving resolution mismatches when saving combined echograms."""
    FINEST = "finest"        # Use finest resolution (highest quality, larger files)
    COARSEST = "coarsest"    # Use coarsest resolution (smallest files, some data loss)
    SPECIFIED = "specified"  # User provides explicit resolution
    AUTO = "auto"            # Adaptive: use median resolution as compromise


@dataclass
class StorageAxisMode:
    """Describes the coordinate system of stored echogram data.
    
    This metadata tells backends how to interpret stored array indices
    and how to transform data for display in different coordinate systems.
    
    X-axis (ping dimension):
    - Can be irregular (one value per ping, default) or regular (fixed resolution grid)
    - Irregular preserves original ping timing
    - Regular enables more efficient storage and combination
    
    Y-axis (sample dimension):
    - Always regular grid within each ping (current behavior)
    - Resolution and origin define the mapping: y_value = y_origin + y_index * y_resolution
    
    Attributes:
        x_axis: Type of x-axis coordinate ("ping_index" or "ping_time")
        y_axis: Type of y-axis coordinate ("sample_index", "sample_nr", "depth", "range")
        x_resolution: X resolution (None = irregular, one per ping; float = regular grid)
        x_origin: Origin for regular x grid (only used if x_resolution is set)
        y_resolution: Y resolution (always regular, default 1.0 for sample_index)
        y_origin: Y origin value (default 0.0)
    """
    
    x_axis: str = "ping_index"
    y_axis: str = "sample_index"
    
    # X-axis: None = irregular (one value per ping), float = regular grid
    x_resolution: Optional[float] = None
    x_origin: Optional[float] = None
    
    # Y-axis: always regular grid
    y_resolution: float = 1.0
    y_origin: float = 0.0
    
    def __post_init__(self):
        """Validate axis types."""
        valid_x = [e.value for e in XAxisType]
        valid_y = [e.value for e in YAxisType]
        
        if self.x_axis not in valid_x:
            raise ValueError(f"x_axis must be one of {valid_x}, got '{self.x_axis}'")
        if self.y_axis not in valid_y:
            raise ValueError(f"y_axis must be one of {valid_y}, got '{self.y_axis}'")
        
        # If x_resolution is set, x_origin must also be set
        if self.x_resolution is not None and self.x_origin is None:
            raise ValueError("x_origin must be set when x_resolution is specified")
    
    @property
    def is_x_regular(self) -> bool:
        """Whether x-axis is stored on a regular grid."""
        return self.x_resolution is not None
    
    @property
    def is_default(self) -> bool:
        """Whether this is the default storage mode (ping_index, sample_index)."""
        return (
            self.x_axis == "ping_index" and 
            self.y_axis == "sample_index" and
            self.x_resolution is None and
            self.y_resolution == 1.0 and
            self.y_origin == 0.0
        )
    
    @classmethod
    def default(cls) -> "StorageAxisMode":
        """Create default storage mode: (ping_index, sample_index).
        
        This matches current behavior where data is stored as raw ping/sample indices.
        """
        return cls(x_axis="ping_index", y_axis="sample_index")
    
    @classmethod
    def ping_time_depth(
        cls, 
        y_resolution: float = 0.1,
        y_origin: float = 0.0,
        x_resolution: Optional[float] = None,
        x_origin: Optional[float] = None,
    ) -> "StorageAxisMode":
        """Create storage mode for ping_time/depth coordinates.
        
        Args:
            y_resolution: Depth resolution in meters (default 0.1m = 10cm).
            y_origin: Starting depth in meters (default 0.0).
            x_resolution: Time resolution in seconds (None = irregular).
            x_origin: Start timestamp (required if x_resolution is set).
        """
        return cls(
            x_axis="ping_time",
            y_axis="depth",
            x_resolution=x_resolution,
            x_origin=x_origin,
            y_resolution=y_resolution,
            y_origin=y_origin,
        )
    
    @classmethod
    def ping_time_range(
        cls,
        y_resolution: float = 0.1,
        y_origin: float = 0.0,
        x_resolution: Optional[float] = None,
        x_origin: Optional[float] = None,
    ) -> "StorageAxisMode":
        """Create storage mode for ping_time/range coordinates.
        
        Args:
            y_resolution: Range resolution in meters (default 0.1m).
            y_origin: Starting range in meters (default 0.0).
            x_resolution: Time resolution in seconds (None = irregular).
            x_origin: Start timestamp (required if x_resolution is set).
        """
        return cls(
            x_axis="ping_time",
            y_axis="range",
            x_resolution=x_resolution,
            x_origin=x_origin,
            y_resolution=y_resolution,
            y_origin=y_origin,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "x_resolution": self.x_resolution,
            "x_origin": self.x_origin,
            "y_resolution": self.y_resolution,
            "y_origin": self.y_origin,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageAxisMode":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            x_axis=data.get("x_axis", "ping_index"),
            y_axis=data.get("y_axis", "sample_index"),
            x_resolution=data.get("x_resolution"),
            x_origin=data.get("x_origin"),
            y_resolution=data.get("y_resolution", 1.0),
            y_origin=data.get("y_origin", 0.0),
        )
    
    def __repr__(self) -> str:
        parts = [f"x={self.x_axis}", f"y={self.y_axis}"]
        if self.is_x_regular:
            parts.append(f"x_res={self.x_resolution}")
        if self.y_resolution != 1.0:
            parts.append(f"y_res={self.y_resolution}")
        if self.y_origin != 0.0:
            parts.append(f"y_origin={self.y_origin}")
        return f"StorageAxisMode({', '.join(parts)})"


def compute_resolution_from_backends(
    backends: list,
    axis: str,
    strategy: ResolutionStrategy = ResolutionStrategy.AUTO,
    specified_resolution: Optional[float] = None,
) -> Optional[float]:
    """Compute resolution for combining multiple backends.
    
    Args:
        backends: List of EchogramDataBackend instances.
        axis: "x" or "y".
        strategy: Resolution strategy to use.
        specified_resolution: Resolution value for SPECIFIED strategy.
        
    Returns:
        Computed resolution, or None if cannot be determined.
    """
    if strategy == ResolutionStrategy.SPECIFIED:
        if specified_resolution is None:
            raise ValueError("specified_resolution required for SPECIFIED strategy")
        return specified_resolution
    
    # Collect resolutions from backends
    resolutions = []
    for backend in backends:
        storage_mode = getattr(backend, 'storage_mode', StorageAxisMode.default())
        
        if axis == "x":
            if storage_mode.is_x_regular:
                resolutions.append(storage_mode.x_resolution)
            else:
                # Estimate from ping times
                ping_times = backend.ping_times
                if len(ping_times) > 1:
                    diffs = np.diff(ping_times)
                    resolutions.append(np.median(diffs[diffs > 0]))
        else:  # y axis
            resolutions.append(storage_mode.y_resolution)
    
    resolutions = [r for r in resolutions if r is not None and np.isfinite(r) and r > 0]
    
    if not resolutions:
        return None
    
    if strategy == ResolutionStrategy.FINEST:
        return min(resolutions)
    elif strategy == ResolutionStrategy.COARSEST:
        return max(resolutions)
    else:  # AUTO
        return float(np.median(resolutions))
