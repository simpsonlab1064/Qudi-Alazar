__all__ = ["DownsampleWorker"]

from PySide2 import QtCore
import numpy as np
import numpy.typing as npt

"""
So far, all algorithms for averaging / windowing are unnacceptably slow.
Might need to write a custom external function for this?

Tried:
numpy.stride_tricks.sliding_window_view
scikit.signal -> decimate
np.convolve
"""


class DownsampleWorker(QtCore.QObject):
    sigDownsampleFinished = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()

    @QtCore.Slot(np.ndarray, int, object)  # type: ignore
    def downsample(
        self, data: npt.NDArray[np.float_], downsample: int, limits: list[float]
    ):
        # Manually clip the data so pg doesn't have to do it in the main thread:
        start = np.argmin(np.abs(data - limits[0]))
        end = np.argmin(np.abs(data - limits[1]))

        assert start <= end

        num_pts = len(data[0, :])

        start = start - 1000 if start > 1000 else 0
        end = end + 1000 if end < len(data[0, :]) - 1000 else num_pts

        # We keep the first and last points in the data set so if someone uses
        # the auto-resize button, it knows the full range
        if downsample > 1:
            x1 = data[0, start:end:downsample]
            x = np.zeros(len(x1) + 2)
            x[1:-1] = x1
            x[0] = data[0, 0]
            x[-1] = data[0, -1]

            y1 = data[1, start:end:downsample]
            y = np.zeros(len(y1) + 2)
            y[1:-1] = y1
            y[0] = data[1, 0]
            y[-1] = data[1, -1]

        else:
            x = data[0, start:end]
            y = data[1, start:end]

        self.sigDownsampleFinished.emit(np.vstack((x, y)))  # type: ignore
