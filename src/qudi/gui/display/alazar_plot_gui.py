__all__ = ["AlazarPlotGui"]

import numpy as np
import numpy.typing as npt

from qudi.core.module import GuiBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from qudi.util.colordefs import QudiPalettePale as palette  # type: ignore
from qudi.logic.base_alazar_logic import (
    BaseExperimentSettings,
    DisplayData,
    DisplayType,
)
from processing_functions.gui.downsample_worker import DownsampleWorker

from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg  # type: ignore

pg.setConfigOption("useOpenGL", True)  # type: ignore


class AlazarPlotDisplayWindow(QtWidgets.QMainWindow):
    sigSelectChannel = QtCore.Signal(int)
    sigDownsample = QtCore.Signal(np.ndarray, int)  # Signal to worker thread
    _data: list[DisplayData]
    _selected_idx: int = 0
    _settings: BaseExperimentSettings
    _initial_downsampling: int = 1000
    _current_downsampling: int = 1000
    _needs_resample: bool = False
    # TODO: Possibly add a flag/check to see if we should discard the downsample result (when we change data)

    def __init__(self, settings: BaseExperimentSettings, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.setWindowTitle("Alazar Plot Display")

        self._data = []
        self._settings = settings

        # Worker and thread for handling downsampling
        self._worker_thread = QtCore.QThread()
        self._worker = DownsampleWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker.sigDownsampleFinished.connect(  # type: ignore
            self._plot_data, QtCore.Qt.QueuedConnection
        )
        self.sigDownsample.connect(self._worker.downsample, QtCore.Qt.QueuedConnection)  # type: ignore

        self._worker_thread.start()

        # Timer to check for updates:
        self.__timer = QtCore.QTimer(self)
        self.__timer.setInterval(500)
        self.__timer.start()
        self.__timer.timeout.connect(self.__resample)  # type: ignore

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAntialiasing(False)  # type: ignore
        self.plot_widget.getPlotItem().setContentsMargins(1, 1, 1, 1)  # type: ignore
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,  # type: ignore
            QtWidgets.QSizePolicy.Expanding,  # type: ignore
        )
        self.plot_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.plot_widget.setLabel("bottom", text="Time", units="ns")
        self.plot_widget.setLabel("left", text="Signal")

        self._data_item = pg.PlotDataItem(
            pen=pg.mkPen(palette.c1, style=QtCore.Qt.SolidLine),  # type: ignore
            autoDownsample=False,
            clipToView=True,
        )

        self.plot_widget.addItem(self._data_item)  # type: ignore

        self.plot_widget.sigXRangeChanged.connect(self._flag_resample)  # type: ignore

        self.data_selection = QtWidgets.QComboBox()
        self.update_data(self._data)
        self.data_selection.currentIndexChanged.connect(self._select_data)  # type: ignore

        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.plot_widget, 0, 0, 4, 4)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.data_selection)
        control_layout.addStretch()

        layout.addLayout(control_layout, 0, 5, 5, 1)
        layout.setColumnStretch(1, 1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self._worker_thread.requestInterruption()
        self._worker_thread.quit()
        self._worker_thread.wait()
        event.accept()

    def update_data(self, data: list[DisplayData]):
        self._data = data

        if not self.data_selection.count() == len(data):
            self.data_selection.clear()
            self._selected_idx = 0

            for i, e in enumerate(self._data):
                if e.type == DisplayType.LINE:
                    self.data_selection.insertItem(i, e.label, e.label)

        else:
            # Assume the channels haven't changed meaning if they are of the
            # same length (good assumption?)
            if len(self._data) > 0:
                self._select_data(self._selected_idx)

        if self.data_selection.count() > self._selected_idx:
            self._select_data(self._selected_idx)

    # @QtCore.Slot(pg.ViewBox, list[list[float]], list[bool])  # type: ignore
    @QtCore.Slot(object, object)  # type: ignore
    def _flag_resample(
        self,
        _: pg.ViewBox,
        limits: list[float],
    ):
        if len(self._data) > 0:
            x_min = limits[0]
            x_max = limits[1]
            points_in_range = round(
                (x_max - x_min) / (self._settings.sample_rate * 1e-9)
            )
            print(points_in_range)
            print(points_in_range < 1e7)

            # Note, you need to go from lowest to highest or it instantly
            # matches on the first choice
            if points_in_range < 1e3:
                self._current_downsampling = 1

            elif points_in_range < 1e5:
                self._current_downsampling = 5

            elif points_in_range < 1e9:
                self._current_downsampling = 50  # disable

            else:
                self._current_downsampling = self._initial_downsampling

            print(f"Downsampling: {self._current_downsampling}")
            self._needs_resample = True

    def _select_data(self, idx: int):
        self._selected_idx = idx
        self._needs_resample = True
        self._current_downsampling = self._initial_downsampling
        self.__resample()

    @QtCore.Slot(np.ndarray)  # type: ignore
    def _plot_data(self, data: npt.NDArray[np.float_]):
        self._data_item.setData(x=data[0], y=data[1])  # type: ignore

    def __resample(self):
        if self._needs_resample:
            ns_per_sample = 1e9 / self._settings.sample_rate
            num_samples = len(self._data[self._selected_idx].data)
            x = np.linspace(start=0, stop=num_samples * ns_per_sample, num=num_samples)
            data = np.vstack((x, self._data[self._selected_idx].data))
            downsample = self._current_downsampling
            self.sigDownsample.emit(data, downsample)  # type: ignore
            self._needs_resample = False


class AlazarPlotGui(GuiBase):
    """
    This GUI displays anything that it thinks is "image-like". It expects data
    to be provided as a list[np.ndarray] and will give the option to plot any
    data that is
    """

    _logic = Connector(name="alazar_logic", interface="BaseAlazarLogic")

    _data: list[DisplayData] = []
    _settings: BaseExperimentSettings

    def on_activate(self):
        self._settings: BaseExperimentSettings = self._logic().experiment_info
        self._mw = AlazarPlotDisplayWindow(self._settings)

        self._logic().sigImageDataUpdated.connect(self._update_data)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    @QtCore.Slot(object)  # type: ignore
    def _update_data(self, data: list[DisplayData]):
        self._data.clear()
        for d in data:
            if d.type == DisplayType.LINE:
                self._data.append(d)

        self._mw.update_data(self._data)
