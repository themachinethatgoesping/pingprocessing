"""VideoFrames – container for captured viewer frames with export helpers."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class VideoFrames:
    """Container for captured video frames with per-frame metadata.

    Stores RGB numpy arrays together with timestamps so that export
    can use either a fixed frame rate or timing derived from the
    actual ping timestamps.

    Examples
    --------
    >>> viewer.frames.export_avif("out.avif", fps=10)
    >>> viewer.frames.export_mp4("out.mp4", fps=25)
    >>> viewer.frames.export_avif("out.avif", ping_time_speed=3.0)
    """

    def __init__(self) -> None:
        self._frames: List[np.ndarray] = []            # RGB uint8 arrays
        self._timestamps: List[Optional[float]] = []   # unix timestamps per frame

    # -- mutation ----------------------------------------------------------

    def clear(self) -> None:
        """Remove all stored frames."""
        self._frames.clear()
        self._timestamps.clear()

    def append(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Append a single RGB frame with optional ping timestamp."""
        self._frames.append(frame)
        self._timestamps.append(timestamp)

    # -- properties --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._frames[idx]

    @property
    def frames(self) -> List[np.ndarray]:
        """All stored RGB frames."""
        return self._frames

    @property
    def timestamps(self) -> List[Optional[float]]:
        """Per-frame timestamps (may contain None)."""
        return self._timestamps

    # -- timing helpers ----------------------------------------------------

    def _compute_durations(self, speed: float = 1.0) -> List[float]:
        """Compute per-frame durations from ping timestamps.

        Parameters
        ----------
        speed : float
            Speed multiplier applied to the real time gaps.
            ``speed=3`` means 3× real-time.

        Returns
        -------
        list of float
            Duration in seconds for each frame transition.
        """
        durations: List[float] = []
        for i in range(1, len(self._timestamps)):
            t_prev = self._timestamps[i - 1]
            t_cur = self._timestamps[i]
            if t_prev is not None and t_cur is not None:
                dt = abs(t_cur - t_prev) / max(speed, 0.001)
                durations.append(max(0.01, dt))
            else:
                durations.append(0.1)  # fallback 100 ms
        return durations

    # -- export ------------------------------------------------------------

    def export_avif(
        self,
        filename: str = "video.avif",
        fps: Optional[float] = None,
        ping_time_speed: Optional[float] = None,
        quality: int = 75,
        loop: int = 0,
    ) -> str:
        """Export frames as animated AVIF.

        Parameters
        ----------
        filename : str
            Output path.
        fps : float, optional
            Fixed frame rate.  Ignored when *ping_time_speed* is set.
        ping_time_speed : float, optional
            Use real ping timestamps scaled by this speed factor
            (e.g. 3.0 = 3× real-time).
        quality : int
            AVIF quality 1–100.
        loop : int
            Number of loops (0 = infinite).

        Returns
        -------
        str
            The filename that was written.
        """
        if len(self._frames) == 0:
            raise ValueError("No frames to export")

        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            raise ImportError("pip install pillow-avif-plugin")
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in self._frames]

        if ping_time_speed is not None:
            durations = self._compute_durations(speed=ping_time_speed)
            duration_ms: Any = [int(d * 1000) for d in durations]
            # first frame needs a duration too
            duration_ms.insert(0, duration_ms[0] if duration_ms else 100)
        elif fps is not None:
            duration_ms = int(1000 / max(fps, 0.1))
        else:
            duration_ms = 100  # default 10 fps

        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=loop,
            quality=quality,
        )
        return filename

    def export_mp4(
        self,
        filename: str = "video.mp4",
        fps: Optional[float] = None,
        ping_time_speed: Optional[float] = None,
        codec: str = "libx264",
        quality: int = 8,
    ) -> str:
        """Export frames as MP4 video.

        Parameters
        ----------
        filename : str
            Output path.
        fps : float, optional
            Fixed frame rate.  Ignored when *ping_time_speed* is set.
        ping_time_speed : float, optional
            Use real ping timestamps; the *average* resulting fps is
            passed to ffmpeg (per-frame variable rate is not supported
            by most containers).
        codec : str
            FFmpeg video codec.
        quality : int
            FFmpeg quality parameter.

        Returns
        -------
        str
            The filename that was written.
        """
        if len(self._frames) == 0:
            raise ValueError("No frames to export")

        try:
            import imageio_ffmpeg  # noqa: F401
            import imageio
        except ImportError:
            raise ImportError("pip install imageio imageio-ffmpeg")

        if ping_time_speed is not None:
            durations = self._compute_durations(speed=ping_time_speed)
            avg_dur = sum(durations) / len(durations) if durations else 0.1
            effective_fps = 1.0 / avg_dur if avg_dur > 0 else 10.0
        elif fps is not None:
            effective_fps = max(fps, 0.1)
        else:
            effective_fps = 10.0

        writer = imageio.get_writer(filename, fps=effective_fps, codec=codec, quality=quality)
        for frame in self._frames:
            writer.append_data(frame)
        writer.close()
        return filename

    def __repr__(self) -> str:
        ts = [t for t in self._timestamps if t is not None]
        dt_str = ""
        if len(ts) >= 2:
            total = ts[-1] - ts[0]
            dt_str = f", span={total:.1f}s"
        return f"VideoFrames({len(self._frames)} frames{dt_str})"
