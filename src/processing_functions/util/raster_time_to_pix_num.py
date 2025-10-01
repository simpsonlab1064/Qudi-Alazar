# camryn made this file

import numpy as np
import numpy.typing as npt

# Set to True if your fast axis is serpentine (bidirectional)
_BIDIRECTIONAL = True


def raster_time_to_pix_num(
    t_us: npt.NDArray[np.float64],
    width: int,
    pixel_dwell_us: float,
) -> npt.NDArray[np.int_]:
    """
    Linear (non-sine) mapper for a piezo raster fast axis.
    Maps each sample time -> pixel label: label = line*width + x.
    - width: pixels per line
    - fast_period_ns: time per line in ns
    - phase: treated as fraction of a line (use 0.0 if you don’t care)
    """
    if width <= 0 or pixel_dwell_us <= 0:
        raise ValueError("width and fast_period_ns must be positive")

    assignment = np.floor(t_us / pixel_dwell_us) # this line does stair-steppies up

    assignment[-1] = assignment[-2]

    # real_assignment = np.concatenate([assignment, assignment[::-1]]) # For a triangle

    return assignment.astype(np.int64)
