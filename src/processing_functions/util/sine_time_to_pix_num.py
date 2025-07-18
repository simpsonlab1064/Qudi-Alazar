__all__ = ["sine_time_to_pix_num"]
import numpy as np
import numpy.typing as npt


def sine_time_to_pix_num(
    time: npt.NDArray[np.float_],
    pixels_per_line: int,
    mirror_period: float,
    fast_phase: float,
) -> npt.NDArray[np.int_]:
    """
    Converts an array of times to pixel locations assuming sinusoidal motion of the
    mirror
    """
    pixels_per_line = pixels_per_line - 1

    assignment = np.round(
        pixels_per_line
        * (0.5 * np.cos(2 * np.pi * time / mirror_period + fast_phase) + 0.5)
    )

    assignment = (
        -assignment
        + pixels_per_line
        + (1 + pixels_per_line) * np.floor((time - 1) / mirror_period)
    )

    assignment[-1] = assignment[-2]

    return assignment.astype(int)
