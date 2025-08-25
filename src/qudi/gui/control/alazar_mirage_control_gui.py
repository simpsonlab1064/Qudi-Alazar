# -*- coding: utf-8 -*-

__all__ = ["AlazarMirageControlGui"]
from qudi.core.module import GuiBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from qudi.util.paths import get_artwork_dir  # type: ignore

from qudi.logic.mirage_logic import MirageExperimentSettings, MirageLogic
import os
from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg  # type: ignore

pg.setConfigOption("useOpenGL", True)  # Add this at the top of your file # type: ignore


class MirageControlWindow(QtWidgets.QMainWindow):
    """Main window for mIRage measurement"""

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.setWindowTitle("Alazar mIRage Control")

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
        self.action_close.triggered.connect(self.close)  # type: ignore
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
        self.fast_mirror_phase.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.fast_mirror_phase.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.fast_mirror_phase.setRange(-7, 7)
        self.fast_mirror_phase.setDecimals(4)

        self.pixel_dwell_time_us_label = QtWidgets.QLabel("Pixel Dwell Time (us)")
        self.pixel_dwell_time_us = QtWidgets.QDoubleSpinBox()
        self.pixel_dwell_time_us.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.pixel_dwell_time_us.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.pixel_dwell_time_us.setRange(1, 100000)
        self.pixel_dwell_time_us.setDecimals(4)

        self.ir_pulse_duration_us_label = QtWidgets.QLabel(
            "IR Laser Pulse Duration (us)"
        )
        self.ir_pulse_duration_us = QtWidgets.QDoubleSpinBox()
        self.ir_pulse_duration_us.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.ir_pulse_duration_us.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.ir_pulse_duration_us.setRange(1, 100000)
        self.ir_pulse_duration_us.setDecimals(4)

        self.wavelengths_per_pixel_label = QtWidgets.QLabel("Wavelengths Per Pixel")
        self.wavelengths_per_pixel = QtWidgets.QSpinBox()
        self.wavelengths_per_pixel.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons  # type: ignore
        )
        self.wavelengths_per_pixel.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.wavelengths_per_pixel.setRange(1, 1024)

        self.image_width_label = QtWidgets.QLabel("Image Width (pixels)")
        self.image_width = QtWidgets.QSpinBox()
        self.image_width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.image_width.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.image_width.setRange(1, 1024)

        self.image_height_label = QtWidgets.QLabel("Image Height (pixels)")
        self.image_height = QtWidgets.QSpinBox()
        self.image_height.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.image_height.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.image_height.setRange(1, 1024)

        self.number_frames_label = QtWidgets.QLabel("Number of Frames")
        self.number_frames = QtWidgets.QSpinBox()
        self.number_frames.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # type: ignore
        self.number_frames.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore
        self.number_frames.setRange(1, 10000)

        self.live_process_fn_label = QtWidgets.QLabel("Live Processing Function")
        self.live_process_fn = QtWidgets.QLineEdit()
        self.live_process_fn.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore

        self.end_process_fn_label = QtWidgets.QLabel("End Processing Function")
        self.end_process_fn = QtWidgets.QLineEdit()
        self.end_process_fn.setAlignment(QtCore.Qt.AlignHCenter)  # type: ignore

        self.save_folder_label = QtWidgets.QLabel("Save Folder Path")
        self.save_folder = QtWidgets.QLineEdit()
        self.save_folder.setAlignment(QtCore.Qt.AlignLeft)  # type: ignore

        self.autosave = QtWidgets.QToolButton()
        self.autosave.setCheckable(True)
        self.autosave.setText("Autosave Data?")

        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()

        control_layout = QtWidgets.QVBoxLayout()

        control_layout.addWidget(self.fast_mirror_phase_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.fast_mirror_phase, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(
            self.pixel_dwell_time_us_label,
            0,
            QtCore.Qt.AlignBottom,  # type: ignore
        )
        control_layout.addWidget(self.pixel_dwell_time_us, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(
            self.ir_pulse_duration_us_label,
            0,
            QtCore.Qt.AlignBottom,  # type: ignore
        )
        control_layout.addWidget(self.ir_pulse_duration_us, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(
            self.wavelengths_per_pixel_label,
            0,
            QtCore.Qt.AlignBottom,  # type: ignore
        )
        control_layout.addWidget(self.wavelengths_per_pixel, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.image_width_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.image_width, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.image_height_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.image_height, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.number_frames_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.number_frames, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.live_process_fn_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.live_process_fn, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.end_process_fn_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.end_process_fn, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.save_folder_label, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.save_folder, 0, QtCore.Qt.AlignTop)  # type: ignore
        control_layout.addWidget(self.autosave, 0, QtCore.Qt.AlignBottom)  # type: ignore
        control_layout.addWidget(self.do_live, 0, QtCore.Qt.AlignTop)  # type: ignore

        control_layout.addWidget(self.start_stop_button)

        layout.addLayout(control_layout, 0, 4, 5, 1)

        # Create dummy widget as main widget and set layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class AlazarMirageControlGui(GuiBase):
    """Qudi GUI module Alazar control"""

    _logic = Connector(name="mirage_logic", interface="MirageLogic")

    #### Output Signals ####
    sigStartMeasurement = QtCore.Signal()
    sigStartLiveMeasurement = QtCore.Signal()
    sigStopMeasurement = QtCore.Signal()
    sigUpdateAcquisitionSettings = QtCore.Signal(
        object
    )  # is a MirageExperimentSettings
    sigSaveData = QtCore.Signal()

    def on_activate(self):
        self._running = False
        self._mw = MirageControlWindow()
        logic: MirageLogic = self._logic()

        # Pull stored values to re-populate:
        settings = logic.experiment_info

        self._mw.fast_mirror_phase.setValue(settings.fast_motion_phase)
        self._mw.pixel_dwell_time_us.setValue(settings.pixel_dwell_time_us)
        self._mw.ir_pulse_duration_us.setValue(settings.ir_pulse_period_us)
        self._mw.wavelengths_per_pixel.setValue(settings.wavelengths_per_pixel)
        self._mw.image_height.setValue(settings.height)
        self._mw.image_width.setValue(settings.width)
        self._mw.number_frames.setValue(settings.num_frames)
        self._mw.live_process_fn.setText(
            settings.live_process_function if not None else ""  # type: ignore
        )

        self._mw.end_process_fn.setText(
            settings.end_process_function if not None else ""  # type: ignore
        )

        self._mw.save_folder.setText(settings.autosave_file_path if not None else "")  # type: ignore

        # Connect internal signals:
        self._mw.fast_mirror_phase.valueChanged.connect(self._fast_mirror_update)  # type: ignore
        self._mw.pixel_dwell_time_us.valueChanged.connect(self._pixel_dwell_time_us)  # type: ignore
        self._mw.ir_pulse_duration_us.valueChanged.connect(self._ir_pulse_duration_us)  # type: ignore
        self._mw.wavelengths_per_pixel.valueChanged.connect(self._wavelengths_per_pixel)  # type: ignore
        self._mw.image_height.valueChanged.connect(self._image_height)  # type: ignore
        self._mw.image_width.valueChanged.connect(self._image_width)  # type: ignore
        self._mw.number_frames.valueChanged.connect(self._number_frames)  # type: ignore
        self._mw.live_process_fn.textChanged.connect(self._live_fn)  # type: ignore
        self._mw.end_process_fn.textChanged.connect(self._end_fn)  # type: ignore
        self._mw.save_folder.textChanged.connect(self._save_folder)  # type: ignore
        self._mw.start_stop_button.clicked.connect(self._start_stop_pressed)  # type: ignore

        # And external siganls:
        # Inputs
        logic.sigAcquisitionAborted.connect(self._measurement_finished)  # type: ignore
        logic.sigAcquisitionCompleted.connect(self._measurement_finished)  # type: ignore
        logic.sigAcquisitionStarted.connect(self._measurement_started)  # type: ignore
        logic.sigLiveAcquisitionStarted.connect(self._measurement_started)  # type: ignore
        logic.sigProgressUpdated.connect(self._progress_update)  # type: ignore
        # Outputs
        self.sigStartMeasurement.connect(  # type: ignore
            logic.start_acquisition, QtCore.Qt.QueuedConnection
        )
        self.sigStartLiveMeasurement.connect(  # type: ignore
            logic.start_live_acquisition, QtCore.Qt.QueuedConnection
        )
        self.sigStopMeasurement.connect(  # type: ignore
            logic.stop_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigUpdateAcquisitionSettings.connect(  # type: ignore
            logic.configure_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigSaveData.connect(logic.save_data, QtCore.Qt.QueuedConnection)  # type: ignore

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    def _start_stop_pressed(self):
        if self._running:
            self.sigStopMeasurement.emit()  # type: ignore

        else:
            # This needs to get reset if we were doing live processing
            self._live_fn(self._mw.live_process_fn.text())  # type: ignore
            if self._mw.do_live.isChecked():
                self.sigStartLiveMeasurement.emit()  # type: ignore
            else:
                self.sigStartMeasurement.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    def _measurement_started(self):
        self._mw.statusbar.showMessage("Acquiring...")
        self._mw.start_stop_button.setText("Stop Measurement")
        self._mw.progress_bar.setValue(0)
        self._running = True

    @QtCore.Slot()  # type: ignore
    def _measurement_finished(self):
        self._mw.statusbar.showMessage("Ready")
        self._mw.start_stop_button.setText("Start Measurement")
        self._mw.progress_bar.setValue(0)
        self._running = False

    @QtCore.Slot(float)  # type: ignore
    def _progress_update(self, percent: float):
        self._mw.progress_bar.setValue(round(percent))

    def _fast_mirror_update(self, phase: float):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.fast_motion_phase = phase
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _pixel_dwell_time_us(self, dwell: float):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.pixel_dwell_time_us = dwell
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _ir_pulse_duration_us(self, pulse: float):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.ir_pulse_period_us = pulse
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _wavelengths_per_pixel(self, num: int):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.wavelengths_per_pixel = num
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _image_height(self, h: int):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.height = h
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _image_width(self, w: int):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.width = w
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _number_frames(self, n: int):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.num_frames = n
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _live_fn(self, fn: str):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.live_process_function = fn
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _end_fn(self, fn: str):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.end_process_function = fn
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _save_folder(self, path: str):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.autosave_file_path = path
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore

    def _do_autosave(self, d: bool):
        settings: MirageExperimentSettings = self._logic().experiment_info
        settings.do_autosave = d
        self.sigUpdateAcquisitionSettings.emit(settings)  # type: ignore
