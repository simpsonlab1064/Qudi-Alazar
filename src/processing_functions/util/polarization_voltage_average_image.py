__all__ = ["polarization_voltage_average_image"]

from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate  # type: ignore
import numpy as np
import numpy.typing as npt


def polarization_voltage_average_image(
    data: npt.NDArray[np.int_],
    assignment: npt.NDArray[np.int_],
    h: int,
    w: int,
    polarization_states: int,
) -> npt.NDArray[np.float_]:
    """
    Returns n = polarization_states images, with 0's where no data was collected.
    Output shape is (n, h, w).
    Assumes data is for a single frame
    """
    images = np.zeros((polarization_states, h * w))

    agg = aggregate(assignment, data)  # type: ignore
    pulses_per_pixel = aggregate(assignment, 1, fill_value=-10)  # type: ignore

    for i in range(polarization_states):
        images[i, :] = agg[i] / pulses_per_pixel[i]

    images = np.reshape(images, (polarization_states, h, w))

    return images
