__all__ = ["AlazarChannelGui"]

from qudi.core.module import GuiBase
from qudi.core.connector import Connector

from qudi.interface.alazar_interface import (
    Range,
    Coupling,
    Termination,
    MeasurementType,
    ChannelInfo,
    BoardInfo,
)

from PySide2 import QtCore, QtWidgets


class ChannelTile(QtWidgets.QWidget):
    sigSettingsChanged = QtCore.Signal(object)  # is a ChannelInfo
    _channel: ChannelInfo

    def __init__(self, channel: ChannelInfo, label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._channel = channel

        self._inner_frame = QtWidgets.QGroupBox(title=label)
        font = self._inner_frame.font()
        font.setBold(True)
        self._inner_frame.setFont(font)

        # self._inner_frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        # self._inner_frame.setLineWidth(1)

        chan = QtWidgets.QVBoxLayout()

        # self._name = QtWidgets.QLabel(label)
        # font = self._name.font()
        # font.setBold(True)
        # self._name.setFont(font)

        self._enabled = QtWidgets.QToolButton()
        self._enabled.setText("Enabled")
        self._enabled.setCheckable(True)
        self._enabled.setChecked(channel.enabled)

        self._range_label = QtWidgets.QLabel("Range")
        self._range = QtWidgets.QComboBox()
        for i, e in enumerate(Range.__members__.items()):
            self._range.insertItem(i, e[0], e[1])

        self._range.setCurrentIndex(self._range.findData(channel.range))

        termination_widget = QtWidgets.QWidget()
        termination_layout = QtWidgets.QHBoxLayout()

        termination_label = QtWidgets.QLabel("Termination")
        self._term_50_ohm = QtWidgets.QToolButton()
        self._term_50_ohm.setText("50 Ohm")
        self._term_50_ohm.setCheckable(True)
        self._term_50_ohm.setAutoExclusive(True)
        self._term_50_ohm.setChecked(channel.termination == Termination.OHM_50)

        self._term_1_mohm = QtWidgets.QToolButton()
        self._term_1_mohm.setText("1 MOhm")
        self._term_1_mohm.setCheckable(True)
        self._term_1_mohm.setAutoExclusive(True)
        self._term_1_mohm.setChecked(channel.termination == Termination.OHM_1M)

        termination_layout.addWidget(self._term_50_ohm)
        termination_layout.addWidget(self._term_1_mohm)

        termination_widget.setLayout(termination_layout)

        coupling_widget = QtWidgets.QWidget()
        coupling_layout = QtWidgets.QHBoxLayout()

        coupling_label = QtWidgets.QLabel("Coupling")
        self._coupling_ac = QtWidgets.QToolButton()
        self._coupling_ac.setText("AC")
        self._coupling_ac.setCheckable(True)
        self._coupling_ac.setAutoExclusive(True)
        self._coupling_ac.setChecked(channel.coupling == Coupling.AC)

        self._coupling_dc = QtWidgets.QToolButton()
        self._coupling_dc.setText("DC")
        self._coupling_dc.setCheckable(True)
        self._coupling_dc.setAutoExclusive(True)
        self._coupling_dc.setChecked(channel.coupling == Coupling.DC)

        coupling_layout.addWidget(self._coupling_ac)
        coupling_layout.addWidget(self._coupling_dc)

        coupling_widget.setLayout(coupling_layout)

        measurement_type_label = QtWidgets.QLabel("Measurement Type")

        self._measurement_type = QtWidgets.QComboBox()
        for i, e in enumerate(MeasurementType.__members__.items()):
            self._measurement_type.insertItem(i, e[0], e[1])

        self._measurement_type.setCurrentIndex(
            self._measurement_type.findData(channel.measurement_type)
        )

        # chan.addWidget(self._name)
        chan.addWidget(self._enabled, alignment=QtCore.Qt.AlignHCenter)

        chan.addWidget(self._range_label)
        chan.addWidget(self._range)

        chan.addWidget(termination_label)
        chan.addWidget(termination_widget)

        chan.addWidget(coupling_label)
        chan.addWidget(coupling_widget)

        chan.addWidget(measurement_type_label)
        chan.addWidget(self._measurement_type)

        self._enabled.toggled.connect(self._enabled_change)
        self._range.currentIndexChanged.connect(self._range_change)
        self._measurement_type.currentIndexChanged.connect(self._type_change)

        self._coupling_ac.toggled.connect(self._coupling_change)
        self._term_50_ohm.toggled.connect(self._term_change)

        self._inner_frame.setLayout(chan)

        t = QtWidgets.QVBoxLayout()
        t.addWidget(self._inner_frame)

        self.setLayout(t)

    def _type_change(self, _: int):
        self._channel.measurement_type = self._measurement_type.currentData()
        self.sigSettingsChanged.emit(self._channel)

    def _range_change(self, _: int):
        self._channel.range = self._range.currentData()
        self.sigSettingsChanged.emit(self._channel)

    def _coupling_change(self, _: bool):
        c = Coupling.AC if self._coupling_ac.isChecked() else Coupling.DC
        self._channel.coupling = c
        self.sigSettingsChanged.emit(self._channel)

    def _term_change(self, _: bool):
        t = Termination.OHM_50 if self._term_50_ohm.isChecked() else Termination.OHM_1M
        self._channel.termination = t
        self.sigSettingsChanged.emit(self._channel)

    def _enabled_change(self, check: bool):
        self._channel.enabled = check
        self.sigSettingsChanged.emit(self._channel)


class BoardPanel(QtWidgets.QWidget):
    sigSettingsChanged = QtCore.Signal(object)  # is a BoardInfo
    _board: BoardInfo
    _panels: list[QtWidgets.QWidget]

    def __init__(self, board: BoardInfo, label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._board = board

        frame = QtWidgets.QGroupBox(title=label)

        chans = QtWidgets.QHBoxLayout()

        for i, c in enumerate(self._board.channels):
            chan = ChannelTile(c, c.label)
            chans.addWidget(chan)

            chan.sigSettingsChanged.connect(
                lambda c, idx=i: self._channel_update(idx, c)
            )

        frame.setLayout(chans)

        t = QtWidgets.QVBoxLayout()
        t.addWidget(frame)

        self.setLayout(t)

        self.setLayout(chans)

    def _channel_update(self, idx: int, chan: ChannelInfo):
        self._board.channels[idx] = chan
        self.sigSettingsChanged.emit(self._board)


class AlazarChannelWindow(QtWidgets.QMainWindow):
    sigSettingsChanged = QtCore.Signal(object)  # is a list[BoardInfo]
    _boards: list[BoardInfo]
    _widgets: list[QtWidgets.QWidget] = []

    def __init__(self, boards: list[BoardInfo], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Alazar Channel Controls")

        self._boards = boards

        layout = QtWidgets.QVBoxLayout()

        for i, b in enumerate(self._boards):
            w = BoardPanel(b, b.label)
            layout.addWidget(w)
            w.sigSettingsChanged.connect(lambda b, idx=i: self._board_update(idx, b))

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _board_update(self, idx: int, board: BoardInfo):
        self._boards[idx] = board
        self.sigSettingsChanged.emit(self._boards)


class AlazarChannelGui(GuiBase):
    _logic = Connector(name="alazar_logic", interface="BaseAlazarLogic")

    # sigBoardSettingsChanged = QtCore.Signal(list[BoardInfo])

    def on_activate(self):
        self._boards: list[BoardInfo] = self._logic().board_info
        self.log.info(f"Length of boards: {len(self._boards)}")
        self._mw = AlazarChannelWindow(self._boards)
        self._mw.sigSettingsChanged.connect(self._logic().update_boards)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()
