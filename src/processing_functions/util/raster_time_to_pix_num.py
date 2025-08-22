# camryn made this file

import numpy as np
import numpy.typing as npt

# Set to True if your fast axis is serpentine (bidirectional)
_BIDIRECTIONAL = True


def raster_time_to_pix_num(
    t_ns: npt.NDArray[np.int_],
    width: int,
    fast_period_ns: float,
    phase: float,
) -> npt.NDArray[np.int_]:
    """
    Linear (non-sine) mapper for a piezo raster fast axis.
    Maps each sample time -> pixel label: label = line*width + x.
    - width: pixels per line
    - fast_period_ns: time per line in ns
    - phase: treated as fraction of a line (use 0.0 if you don’t care)
    """
    if width <= 0 or fast_period_ns <= 0:
        raise ValueError("width and fast_period_ns must be positive")

    t = np.asarray(t_ns, dtype=np.float64)
    period = float(fast_period_ns)

    # interpret 'phase' as fraction of a line; wrap to [0,1)
    phase_frac = float(phase) - np.floor(float(phase))
    tt = t + phase_frac * period

    line = np.floor(tt / period).astype(np.int64)
    frac = (tt / period) - np.floor(tt / period)  # 0..1 within the line
    x = np.floor(frac * width).astype(np.int64)
    x = np.clip(x, 0, width - 1)

    if _BIDIRECTIONAL:
        x = np.where((line & 1) == 1, (width - 1) - x, x)

    labels = line * width + x
    return labels.astype(np.int64)
