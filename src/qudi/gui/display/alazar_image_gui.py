__all__ = ["AlazarImageGui"]

from qudi.core.module import GuiBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore

from qudi.logic.base_alazar_logic import (
    BaseExperimentSettings,
    DisplayData,
    DisplayType,
)

from PySide2 import QtCore, QtWidgets
import pyqtgraph as pg  # type: ignore

pg.setConfigOption("useOpenGL", True)  # type: ignore


class AlazarImageDisplayWindow(QtWidgets.QMainWindow):
    sigSelectChannel = QtCore.Signal(int)
    _data: list[DisplayData]
    _selected_idx: int = 0

    def __init__(self, settings: BaseExperimentSettings, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.setWindowTitle("Alazar Image Display")

        self._data = []

        self.image = pg.ImageView()

        self.data_selection = QtWidgets.QComboBox()
        self.update_data(self._data)
        self.data_selection.currentIndexChanged.connect(self._select_data)  # type: ignore

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

        if not self.data_selection.count() == len(data):
            self.data_selection.clear()
            self._selected_idx = 0

            for i, e in enumerate(self._data):
                if e.type == DisplayType.IMAGE:
                    self.data_selection.insertItem(i, e.label, e.label)

        else:
            # Assume the channels haven't changed meaning if they are of the
            # same length (good assumption?)
            if len(self._data) > 0:
                self._select_data(self._selected_idx)

        if self.data_selection.count() > self._selected_idx:
            self._select_data(self._selected_idx)

    def _select_data(self, idx: int):
        self._selected_idx = idx
        self.image.setImage(self._data[idx].data)  # type: ignore


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
        self._mw = AlazarImageDisplayWindow(self._settings)

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
            if d.type == DisplayType.IMAGE:
                self._data.append(d)

        self._mw.update_data(self._data)
