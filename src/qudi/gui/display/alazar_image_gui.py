__all__ = ["AlazarDisplayGui"]

from qudi.core.module import GuiBase
from qudi.core.connector import Connector

from qudi.logic.base_alazar_logic import BoardInfo, BaseExperimentSettings
from qudi.interface.alazar_interface import MeasurementType, BoardInfo, ChannelInfo

from PySide2 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg

pg.setConfigOption("useOpenGL", True)


class AlazarDisplayWindow(QtWidgets.QMainWindow):
    sigSelectChannel = QtCore.Signal(int)
    _boards: list[BoardInfo]

    def __init__(
        self, boards: list[BoardInfo], settings: BaseExperimentSettings, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Alazar Image Display")

        self._boards = boards

        self.image = pg.ImageView()

        self.channel_selection = QtWidgets.QComboBox()
        self.update_boards(self._boards)

        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.image, 0, 0, 4, 4)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.channel_selection)
        control_layout.addStretch()

        layout.addLayout(control_layout, 0, 5, 5, 1)
        layout.setColumnStretch(1, 1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _update_channels(self, boards: list[BoardInfo]) -> list[ChannelInfo]:
        channels: list[ChannelInfo] = []
        for b in boards:
            for c in b.channels:
                if c.enabled:
                    channels.append(c)

        return channels

    def update_boards(self, boards: list[BoardInfo]):
        self._boards = boards
        self.channel_selection.clear()
        channels: list[ChannelInfo] = self._update_channels(self._boards)

        for i, e in enumerate(channels):
            self.channel_selection.insertItem(i, e.label, e.label)


class AlazarImageGui(GuiBase):
    _logic = Connector(name="alazar_logic", interface="BaseAlazarLogic")

    _boards: list[BoardInfo]
    _data: np.ndarray = np.array([])
    _settings: BaseExperimentSettings
    _selected_channel: tuple[int, int] = (0, 0)

    def on_activate(self):
        self._boards: list[BoardInfo] = self._logic().board_info
        self._settings: BaseExperimentSettings = self._logic().experiment_info
        self._mw = AlazarDisplayWindow(self._boards, self._settings)
        self._mw.channel_selection.currentIndexChanged.connect(self._select_channel)

        self._logic().sigImageDataUpdated.connect(self._update_data)
        self._logic().sigBoardInfo.connect(self._update_boards)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    @QtCore.Slot(object)
    def _update_boards(self, boards: list[BoardInfo]):
        self._boards = boards
        self._mw.update_boards(self._boards)

    @QtCore.Slot(np.ndarray)
    def _update_data(self, data: np.ndarray):
        self._data = data
        self._mw.image.setImage(self._data[self._selected_channel])

    def _select_channel(self, idx: int):
        """Converts from linear index to actual board/channel number"""

        board_idx = 0

        lin_idx = 0
        if idx >= 0:
            for b in self._boards:
                chan_idx = 0
                for c in b.channels:
                    if c.enabled:
                        if lin_idx == idx:
                            self._selected_channel = (board_idx, chan_idx)
                            self._mw.image.setImage(self._data[self._selected_channel])
                            return
                        lin_idx += 1
                    chan_idx += 1
                board_idx += 1

            self.log.warning(
                f"Could not find enabled channel with linear index: {idx} in the system!"
            )
