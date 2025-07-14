__all__ = ["AlazarImageGui"]

from qudi.core.module import GuiBase
from qudi.core.connector import Connector

from qudi.logic.base_alazar_logic import (
    BaseExperimentSettings,
    DisplayData,
    DisplayType,
)

from PySide2 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg

pg.setConfigOption("useOpenGL", True)


class AlazarDisplayWindow(QtWidgets.QMainWindow):
    sigSelectChannel = QtCore.Signal(int)
    _data: list[DisplayData]

    def __init__(self, settings: BaseExperimentSettings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Alazar Image Display")

        self._data = []

        self.image = pg.ImageView()

        self.data_selection = QtWidgets.QComboBox()
        self.update_data(self._data)
        self.data_selection.currentIndexChanged.connect(self._select_data)

        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.image, 0, 0, 4, 4)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.data_selection)
        control_layout.addStretch()

        layout.addLayout(control_layout, 0, 5, 5, 1)
        layout.setColumnStretch(1, 1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def update_data(self, data: list[DisplayData]):
        self._data = data
        self.data_selection.clear()

        for i, e in enumerate(self._data):
            if e.type == DisplayType.IMAGE:
                self.data_selection.insertItem(i, e.label, e.label)

    def _select_data(self, idx: int):
        self.image.setImage(self._data[idx].data)


class AlazarImageGui(GuiBase):
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
        self._mw = AlazarDisplayWindow(self._settings)

        self._logic().sigImageDataUpdated.connect(self._update_data)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    @QtCore.Slot(object)
    def _update_data(self, data: list[DisplayData]):
        self._data.clear()
        for d in data:
            if d.type == DisplayType.IMAGE:
                self._data.append(d)

        self._mw.update_data(self._data)
