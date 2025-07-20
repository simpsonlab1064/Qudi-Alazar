__all__ = ["DownsampleWorker"]

from PySide2 import QtCore
import numpy as np
import numpy.typing as npt


class DownsampleWorker(QtCore.QObject):
    sigDownsampleFinished = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()

    @QtCore.Slot(np.ndarray, int)  # type: ignore
    def downsample(self, data: npt.NDArray[np.float_], downsample: int):
        x = data[0, ::downsample]
        y = data[1, ::downsample]

        self.sigDownsampleFinished.emit(np.vstack((x, y)))  # type: ignore
