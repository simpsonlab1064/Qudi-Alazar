__all__ = ["voltage_average_image"]

from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate
import numpy as np


def voltage_average_image(
    data: np.ndarray,
    assignment: np.ndarray,
    pulses_per_pixel: np.ndarray,
    h: int,
    w: int,
    frames: int,
) -> np.ndarray:
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
    frame_data: np.ndarray, assignment: np.ndarray, pulses_per_pixel: np.ndarray
) -> np.ndarray:
    sum_per_pixel = aggregate(assignment, frame_data)

    return np.divide(sum_per_pixel, pulses_per_pixel)
