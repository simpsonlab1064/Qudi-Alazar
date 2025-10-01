__all__ = ["photon_counting_image"]

from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate  # type: ignore
import numpy as np
import numpy.typing as npt


def photon_counting_image(
    data: npt.NDArray[np.int_],
    assignment: npt.NDArray[np.int_],
    pulses_per_pixel: npt.NDArray[np.int_],
    h: int,
    w: int,
    frames: int,
    threshold: int,
) -> npt.NDArray[np.float64]:
    image = np.zeros((h, w))
    samples = len(assignment)

    for frame in range(frames):
        start = frame * samples
        end = (frame + 1) * samples
        frame_data = data[start:end]
        im_data = _photon_counting(
            frame_data,
            assignment,
            pulses_per_pixel,
            threshold,
        )
        im_data = im_data.reshape((w, h))
        image += im_data

    return image / frames


def _photon_counting(
    frame_data: npt.NDArray[np.int_],
    assignment: npt.NDArray[np.int_],
    pulses_per_pixel: npt.NDArray[np.int_],
    threshold: int,
) -> npt.NDArray[np.int_]:
    counts = frame_data < threshold
    counts = counts.astype(int)
    sum_per_pixel = aggregate(assignment, counts)  # type: ignore

    return np.divide(sum_per_pixel, pulses_per_pixel)  # type: ignore
