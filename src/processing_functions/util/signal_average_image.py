# camryn made this file

import numpy as np
import numpy.typing as npt
from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate  # type: ignore
from typing import Any, cast

def signal_average_image(
    signal: npt.NDArray[np.generic],
    labels: npt.NDArray[np.int_],
    counts: npt.NDArray[np.int_],
    height: int,
    width: int,
    *_ignore: Any,
) -> npt.NDArray[np.float64]:
    """
    Generic bin + average into a (width, height) image.
    Works for photodiode voltage/current/etc.
    - labels must be: y*width + x (what the mapper returns)
    - counts should be aggregate(labels, 1) from your caller
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    sig = np.asarray(signal, dtype=np.float64).ravel()
    lbl = np.asarray(labels, dtype=np.int64).ravel()
    if sig.size != lbl.size:
        raise ValueError("signal and labels must be the same length")

    summed = np.asarray(cast(npt.NDArray[np.float64], aggregate(lbl, sig)), dtype=np.float64)

    cnt = np.asarray(counts, dtype=np.float64).ravel()
    # pad/truncate counts to match summed length (safety)
    if cnt.shape[0] < summed.shape[0]:
        cnt = np.pad(cnt, (0, summed.shape[0] - cnt.shape[0]), constant_values=0)
    elif cnt.shape[0] > summed.shape[0]:
        cnt = cnt[:summed.shape[0]]

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = summed / np.maximum(cnt, 1.0)

    img = np.zeros((width, height), dtype=np.float64)
    total_bins = avg.shape[0]
    lines = min(height, total_bins // width)
    if lines > 0:
        img[:, :lines] = avg[: lines * width].reshape(lines, width).T  # (w, lines)
    return img
