__all__ = ["voltage_average_image"]

from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate  # type: ignore
import numpy as np
import numpy.typing as npt


def voltage_average_image(
    data: npt.NDArray[np.int_],
    assignment: npt.NDArray[np.int_],
    pulses_per_pixel: npt.NDArray[np.int_],
    h: int,
    w: int,
    frames: int,
) -> npt.NDArray[np.float64]:
    image = np.zeros((h, w))
    samples = len(assignment)

    for frame in range(frames):
        start = frame * samples
        end = (frame + 1) * samples
        frame_data = data[start:end]
        im_data = _voltage_average(frame_data, assignment, pulses_per_pixel)
        im_data = im_data.reshape((w, h))
        image += im_data

    return image / frames


def _voltage_average(
    frame_data: npt.NDArray[np.int_],
    assignment: npt.NDArray[np.int_],
    pulses_per_pixel: npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
    sum_per_pixel = aggregate(assignment, frame_data)  # type: ignore

    return np.divide(sum_per_pixel, pulses_per_pixel)  # type: ignore
