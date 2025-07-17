# -*- coding: utf-8 -*-

__all__ = ["AlazarGalvoResControlGui"]
from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from qudi.util.paths import get_artwork_dir

from qudi.logic.galvo_res_logic import GalvoResExperimentSettings, GalvoResLogic
import os
from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

pg.setConfigOption("useOpenGL", True)  # Add this at the top of your file


class GalvoResControlWindow(QtWidgets.QMainWindow):
    """Main window for Grating Scan measurement"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Alazar Galvo-Res Control")

        # Create menu bar
        menu_bar = QtWidgets.QMenuBar()
        menu = menu_bar.addMenu("File")
        self.action_save_data = QtWidgets.QAction("Save Data")
        path = os.path.join(get_artwork_dir(), "icons", "document-save")
        self.action_save_data.setIcon(QtGui.QIcon(path))
        menu.addAction(self.action_save_data)
        menu.addSeparator()

        self.action_close = QtWidgets.QAction("Close")
        path = os.path.join(get_artwork_dir(), "icons", "application-exit")
        self.action_close.setIcon(QtGui.QIcon(path))
        self.action_close.triggered.connect(self.close)
        menu.addAction(self.action_close)
        self.setMenuBar(menu_bar)

        # Create statusbar and indicators
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.statusbar.addWidget(self.progress_bar)

        # Initialize widgets
        self.start_stop_button = QtWidgets.QPushButton("Start Measurement")

        self.do_live = QtWidgets.QToolButton()
        self.do_live.setCheckable(True)
        self.do_live.setText("Live Viewing?")

        self.fast_mirror_phase_label = QtWidgets.QLabel("Fast Mirror Phase")
        self.fast_mirror_phase = QtWidgets.QDoubleSpinBox()
        self.fast_mirror_phase.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.fast_mirror_phase.setAlignment(QtCore.Qt.AlignHCenter)
        self.fast_mirror_phase.setRange(-7, 7)
        self.fast_mirror_phase.setDecimals(4)

        self.scan_period_us_label = QtWidgets.QLabel("Scan Period (us)")
        self.scan_period_us = QtWidgets.QDoubleSpinBox()
        self.scan_period_us.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.scan_period_us.setAlignment(QtCore.Qt.AlignHCenter)
        self.scan_period_us.setRange(0, 1000)
        self.scan_period_us.setDecimals(4)

        self.image_width_label = QtWidgets.QLabel("Image Width (pixels)")
        self.image_width = QtWidgets.QSpinBox()
        self.image_width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.image_width.setAlignment(QtCore.Qt.AlignHCenter)
        self.image_width.setRange(1, 1024)

        self.image_height_label = QtWidgets.QLabel("Image Height (pixels)")
        self.image_height = QtWidgets.QSpinBox()
        self.image_height.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.image_height.setAlignment(QtCore.Qt.AlignHCenter)
        self.image_height.setRange(1, 1024)

        self.number_frames_label = QtWidgets.QLabel("Number of Frames")
        self.number_frames = QtWidgets.QSpinBox()
        self.number_frames.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.number_frames.setAlignment(QtCore.Qt.AlignHCenter)
        self.number_frames.setRange(1, 10000)

        self.series_length_label = QtWidgets.QLabel("Series Length")
        self.series_length = QtWidgets.QSpinBox()
        self.series_length.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.series_length.setAlignment(QtCore.Qt.AlignHCenter)
        self.series_length.setRange(1, 10000)

        self.do_series = QtWidgets.QToolButton()
        self.do_series.setCheckable(True)
        self.do_series.setText("Do Series?")

        self.live_process_fn_label = QtWidgets.QLabel("Live Processing Function")
        self.live_process_fn = QtWidgets.QLineEdit()
        self.live_process_fn.setAlignment(QtCore.Qt.AlignHCenter)

        self.end_process_fn_label = QtWidgets.QLabel("End Processing Function")
        self.end_process_fn = QtWidgets.QLineEdit()
        self.end_process_fn.setAlignment(QtCore.Qt.AlignHCenter)

        self.save_folder_label = QtWidgets.QLabel("Save Folder Path")
        self.save_folder = QtWidgets.QLineEdit()
        self.save_folder.setAlignment(QtCore.Qt.AlignLeft)

        self.autosave = QtWidgets.QToolButton()
        self.autosave.setCheckable(True)
        self.autosave.setText("Autosave Data?")

        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()

        control_layout = QtWidgets.QVBoxLayout()

        control_layout.addWidget(self.fast_mirror_phase_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.fast_mirror_phase, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.scan_period_us_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.scan_period_us, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.image_width_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.image_width, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.image_height_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.image_height, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.number_frames_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.number_frames, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.series_length_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.series_length, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.do_series, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.live_process_fn_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.live_process_fn, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.end_process_fn_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.end_process_fn, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.save_folder_label, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.save_folder, 0, QtCore.Qt.AlignTop)
        control_layout.addWidget(self.autosave, 0, QtCore.Qt.AlignBottom)
        control_layout.addWidget(self.do_live, 0, QtCore.Qt.AlignTop)

        control_layout.addWidget(self.start_stop_button)

        layout.addLayout(control_layout, 0, 4, 5, 1)

        # Create dummy widget as main widget and set layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class AlazarGalvoResControlGui(GuiBase):
    """Qudi GUI module Alazar control"""

    _logic = Connector(name="galvo_res_logic", interface="GalvoResLogic")

    #### Output Signals ####
    sigStartMeasurement = QtCore.Signal()
    sigStartLiveMeasurement = QtCore.Signal()
    sigStopMeasurement = QtCore.Signal()
    sigUpdateAcquisitionSettings = QtCore.Signal(
        object
    )  # is a GalvoResExperimentSettings
    sigSaveData = QtCore.Signal()

    def on_activate(self):
        self._running = False
        self._mw = GalvoResControlWindow()
        logic: GalvoResLogic = self._logic()

        # Pull stored values to re-populate:
        settings = logic.experiment_info

        self._mw.fast_mirror_phase.setValue(settings.fast_mirror_phase)
        self._mw.scan_period_us.setValue(settings.mirror_period_us)
        self._mw.image_height.setValue(settings.height)
        self._mw.image_width.setValue(settings.width)
        self._mw.number_frames.setValue(settings.num_frames)
        self._mw.series_length.setValue(settings.series_length)
        self._mw.live_process_fn.setText(
            settings.live_process_function if not None else ""
        )

        self._mw.end_process_fn.setText(
            settings.end_process_function if not None else ""
        )

        self._mw.save_folder.setText(settings.autosave_file_path if not None else "")

        # Connect internal signals:
        self._mw.fast_mirror_phase.valueChanged.connect(self._fast_mirror_update)
        self._mw.scan_period_us.valueChanged.connect(self._scan_period_us)
        self._mw.image_height.valueChanged.connect(self._image_height)
        self._mw.image_width.valueChanged.connect(self._image_width)
        self._mw.number_frames.valueChanged.connect(self._number_frames)
        self._mw.series_length.valueChanged.connect(self._series_length)
        self._mw.do_series.toggled.connect(self._do_series)
        self._mw.live_process_fn.textChanged.connect(self._live_fn)
        self._mw.end_process_fn.textChanged.connect(self._end_fn)
        self._mw.save_folder.textChanged.connect(self._save_folder)
        self._mw.autosave.toggled.connect(self._do_series)
        self._mw.start_stop_button.clicked.connect(self._start_stop_pressed)

        # And external siganls:
        # Inputs
        logic.sigAcquisitionAborted.connect(self._measurement_finished)
        logic.sigAcquisitionCompleted.connect(self._measurement_finished)
        logic.sigAcquisitionStarted.connect(self._measurement_started)
        logic.sigLiveAcquisitionStarted.connect(self._measurement_started)
        logic.sigProgressUpdated.connect(self._progress_update)
        # Outputs
        self.sigStartMeasurement.connect(
            logic.start_acquisition, QtCore.Qt.QueuedConnection
        )
        self.sigStartLiveMeasurement.connect(
            logic.start_live_acquisition, QtCore.Qt.QueuedConnection
        )
        self.sigStopMeasurement.connect(
            logic.stop_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigUpdateAcquisitionSettings.connect(
            logic.configure_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigSaveData.connect(logic.save_data, QtCore.Qt.QueuedConnection)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    def _start_stop_pressed(self):
        if self._running:
            self.sigStopMeasurement.emit()

        else:
            if self._mw.do_live.isChecked():
                self.sigStartLiveMeasurement.emit()
            else:
                self.sigStartMeasurement.emit()

    @QtCore.Slot()
    def _measurement_started(self):
        self._mw.statusbar.showMessage("Acquiring...")
        self._mw.start_stop_button.setText("Stop Measurement")
        self._mw.progress_bar.setValue(0)
        self._running = True

    @QtCore.Slot()
    def _measurement_finished(self):
        self._mw.statusbar.showMessage("Ready")
        self._mw.start_stop_button.setText("Start Measurement")
        self._mw.progress_bar.setValue(0)
        self._running = False

    @QtCore.Slot(float)
    def _progress_update(self, percent: float):
        self._mw.progress_bar.setValue(round(percent))

    def _fast_mirror_update(self, phase: float):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.fast_mirror_phase = phase
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _scan_period_us(self, period: float):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.scan_period_us = period
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _image_height(self, h: int):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.height = h
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _image_width(self, w: int):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.width = w
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _number_frames(self, n: int):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.num_frames = n
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _series_length(self, n: int):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.series_length = n
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _do_series(self, d: bool):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.do_series = d
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _live_fn(self, fn: str):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.live_process_function = fn
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _end_fn(self, fn: str):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.end_process_function = fn
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _save_folder(self, path: str):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.autosave_file_path = path
        self.sigUpdateAcquisitionSettings.emit(settings)

    def _do_autosave(self, d: bool):
        settings: GalvoResExperimentSettings = self._logic().experiment_info
        settings.do_autosave = d
        self.sigUpdateAcquisitionSettings.emit(settings)
