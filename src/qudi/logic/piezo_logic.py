from __future__ import annotations

__all__ = ["PiezoLogic"]

from PySide2 import QtCore
import numpy as np
import numpy.typing as npt
from logging import Logger
from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from qudi.logic.base_alazar_logic import BaseAlazarLogic
from qudi.logic.experiment_defs import ImagingExperimentSettings
from qudi.interface.alazar_interface import BoardInfo, AcquisitionMode
from qudi.interface.piezo_stage_interface import PiezoScanSettings, PiezoStageInterface


class PiezoExperimentSettings(ImagingExperimentSettings):
    wavelengths_per_pixel: int
    piezo_settings: PiezoScanSettings

    def __init__(
        self,
        fast_mirror_phase: float = 0,
        mirror_period_us: float = 1000,
        wavelengths_per_pixel: int = 1,
        piezo_settings: PiezoScanSettings = PiezoScanSettings(),
        width: int = 512,
        height: int = 512,
        num_frames: int = 1,
        autosave_file_path: str | None = None,
        do_autosave: bool = False,
        live_processing_function: str | None = None,
        end_processing_function: str | None = None,
    ):
        self.width = width
        self.height = height
        self.piezo_settings = piezo_settings
        self.wavelengths_per_pixel = wavelengths_per_pixel

        super().__init__(
            width=width,
            height=height,
            autosave_file_path=autosave_file_path,
            do_autosave=do_autosave,
            live_process_function=live_processing_function,
            end_process_function=end_processing_function,
            fast_motion_phase=fast_mirror_phase,
            fast_motion_period_us=mirror_period_us,
            num_frames=num_frames,
        )

    def calc_records_per_acquisition(self) -> int:
        return self.wavelengths_per_pixel * self.height * self.width

    def scan_freq_hz(self) -> float:
        return 1e6 / self.fast_motion_period_us

    def update_piezo_settings(self, logger: Logger) -> PiezoScanSettings:
        """
        This function modifies self in-place and returns the updated
        version of the [PiezoScanSettings] for passing to the hardware.
        """

        a_wave_on = self.wavelengths_per_pixel
        a_wave_off = 0
        fast_wave_b_pulses = 1

        out_settings = PiezoScanSettings(
            a_wave_on=a_wave_on,
            a_wave_off=a_wave_off,
            fast_wave_b_pulses=fast_wave_b_pulses,
            fast_wave_ramp_steps=self.width,
            fast_wave_scans_per_slow=1,
            slow_wave_ramp_steps=self.height,
            slow_wave_scans_per_trigger=self.num_frames,
            slow_wave_enable_mode=self.piezo_settings.slow_wave_enable_mode,
        )

        self.piezo_settings = out_settings
        return out_settings

    @staticmethod  # The order here is important -- it must match the __init__ order
    def representer_func(instance: PiezoExperimentSettings) -> list[object]:
        return [
            instance.fast_motion_phase,
            instance.fast_motion_period_us,
            instance.wavelengths_per_pixel,
            instance.piezo_settings,  # TODO: check if called recursively (hopefully)
            instance.width,
            instance.height,
            instance.num_frames,
            instance.autosave_file_path,
            instance.do_autosave,
            instance.live_process_function,
            instance.end_process_function,
        ]

    @staticmethod
    def constructor_func(yaml_data: object) -> PiezoExperimentSettings:
        # return PiezoExperimentSettings(*yaml_data)  # type: ignore
        return yaml_data  # type: ignore # TODO: This appears to not store stuff. I don't know why


class PiezoLogic(BaseAlazarLogic[PiezoExperimentSettings]):
    """
    This contains logic for running an experiment using both the mIRage and
    laser

    Example config that goes into the config file:

    example_logic:
        module.Class: 'piezo_logic.PiezoLogic'
        connect:
            alazar: alazar
            piezo_stage: piezo_stage
        options:
            num_buffers: 0 # 0 to use as many buffers as needed to prevent
                           # overflows. However, this might exhaust
                           # RAM, so can set to a number (>= 2) to use that many
                           # buffers and hope we process them fast enough to not
                           # hit an overrun
            min_mod: 1e-12 # Maximum value that samples per line can be different
                           # without triggering an error

            image_at_end: False # if the end_function produces data that is appropriate for imaging

    """

    _fast_motion_period_us: float = ConfigOption(
        name="fast_motion_period_us", default=2500.0, missing="info"
    ) #type: ignore
    _fast_motion_phase: float = ConfigOption(
        name="fast_motion_phase", default=0.0, missing="info"
    ) #type: ignore
    _min_mod: float = ConfigOption(name="min_mod", default=1e-12, missing="info")  # type: ignore

    _settings: PiezoExperimentSettings = StatusVar(
        name="piezo_settings",
        default=PiezoExperimentSettings(),
        constructor=PiezoExperimentSettings.constructor_func,
        representer=PiezoExperimentSettings.representer_func,
    )  # type: ignore

    _piezo_stage = Connector(name="piezo_stage", interface="PiezoStageInterface")

    # Signals:
    sigSettingsUpdated = QtCore.Signal(object)  # is a PiezoScanSettings
    sigStartStage = QtCore.Signal()
    sigStopStage = QtCore.Signal()
    # Track what we intend to start once the stage is armed
    _pending_start_kind: str | None = None  # "normal" or "live"

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        piezo: PiezoStageInterface = self._piezo_stage()

        # Pull up settings from hardware config:
        p_settings = piezo.get_settings
        self._settings.piezo_settings.clk = p_settings.clk
        self._settings.piezo_settings.wave_f_mode = p_settings.wave_f_mode
        self._settings.piezo_settings.enable_polarity = p_settings.enable_polarity

        #### OUTPUTS ####
        self.sigSettingsUpdated.connect(  # type: ignore
            piezo.update_settings, QtCore.Qt.QueuedConnection
        )
        self.sigStartStage.connect(  # type: ignore
            piezo.download_and_arm, QtCore.Qt.QueuedConnection
        )
        self.sigStopStage.connect(  # type: ignore
            piezo.end_scan, QtCore.Qt.QueuedConnection
        )

        #### INPUTS ####
        piezo.sigStageArmed.connect(  # type: ignore
            self._stage_armed
        )

        self._settings.fast_motion_period_us = self._fast_motion_period_us
        self._settings.fast_motion_phase = self._fast_motion_phase

        #### Super Stuff ####
        super().on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()

    @property
    def board_info(self) -> list[BoardInfo]:
        return super().board_info

    @property
    def experiment_info(self) -> PiezoExperimentSettings:
        return super().experiment_info

    @QtCore.Slot(object)  # type: ignore
    def update_boards(self, boards: list[BoardInfo]):
        super().update_boards(boards)

    @QtCore.Slot(np.ndarray)  # type: ignore
    def _new_alazar_data(self, buf: npt.NDArray[np.int_]):
        super()._new_alazar_data(buf)

    # Camryn commented out the 3 lines below and added the 4 right after them to try to get the piezo stage to stop after Alazar acquisition stops
    # @QtCore.Slot()  # type: ignore
    # def _acquisition_completed(self):
    #    super()._acquisition_completed()

    @QtCore.Slot()  # type: ignore
    def _acquisition_completed(self):
        self.sigStopStage.emit()  # type: ignore
        super()._acquisition_completed()

    # TODO: Camryn: you will need to adjust both "start_*" functions to
    # work the way you want. Basically, I don't know how things are triggering
    # each other. My guess is that you'll want to break it into two pieces, one
    # arming the stage and then the second starting Alazar (after the stage)
    # is primed and ready
    # @QtCore.Slot()  # type: ignore
    # def start_acquisition(self):
    # num_b = (
    #    self._settings.calc_records_per_acquisition()
    #    if self._num_buffers == 0
    #    else self._num_buffers
    # )
    # self._apply_configuration(
    #      settings=self._settings,
    #       mode=AcquisitionMode.NPT,  # TODO: Camryn, is this what you want?
    #        num_buffers=num_b,
    #     )
    #      self.sigStartStage.emit()  # TODO: Camryn
    #       super().start_acquisition()
    #
    # @QtCore.Slot(int)  # type: ignore
    # def start_live_acquisition(self):
    # live_settings = self._settings
    # live_settings.num_frames = self._settings.num_frames
    # live_settings.live_process_function = self._live_viewing_fn
    # self._apply_configuration(
    #    settings=live_settings,
    #     mode=AcquisitionMode.NPT,
    #     num_buffers=self._settings.num_frames,
    # )
    # self.sigStartStage.emit()  # TODO: Camryn
    # super().start_live_acquisition()

    @QtCore.Slot()  # type: ignore
    def start_acquisition(self):
        num_b = (
            self._settings.calc_records_per_acquisition()
            if self._num_buffers == 0
            else self._num_buffers
        )
        self._apply_configuration(
            settings=self._settings,
            mode=AcquisitionMode.NPT,  # keep unless you know you need a different mode
            num_buffers=num_b,
        )
        self._pending_start_kind = "normal"
        self.sigStartStage.emit()  # type: ignore  # Stage arms now; Alazar start happens in _stage_armed()

    @QtCore.Slot(int)  # type: ignore
    def start_live_acquisition(self):
        live_settings = self._settings
        live_settings.num_frames = self._settings.num_frames
        live_settings.live_process_function = self._live_viewing_fn
        self._apply_configuration(
            settings=live_settings,
            mode=AcquisitionMode.NPT,
            num_buffers=self._settings.num_frames,
        )
        self._pending_start_kind = "live"
        self.sigStartStage.emit()  # type: ignore   # Stage arms now; Alazar start happens in _stage_armed()

    # @QtCore.Slot()  # type: ignore
    # def stop_acquisition(self):
    # super().stop_acquisition()

    @QtCore.Slot()  # type: ignore
    def stop_acquisition(self):
        # Stop the piezo stage first
        self.sigStopStage.emit()  # type: ignore
        # Then stop the Alazar acquisition
        super().stop_acquisition()

    @QtCore.Slot(object)  # type: ignore
    def configure_acquisition(self, settings: PiezoExperimentSettings):
        p_settings = self._settings.piezo_settings
        super().configure_acquisition(settings)  # This resets all the _settings
        self._settings.piezo_settings = p_settings
        self._settings.fast_motion_period_us = self._fast_motion_period_us
        self._settings.fast_motion_phase = self._fast_motion_phase
        self._settings.update_piezo_settings(logger=self.log)

        # Now we need to update the piezo stage settings and pass those updates
        # to the hardware module:

    @QtCore.Slot()  # type: ignore
    def save_data(self):
        super().save_data()

    def _check_config(self):
        super()._check_config()

    def _apply_configuration(
        self,
        settings: PiezoExperimentSettings,
        mode: AcquisitionMode,
        num_buffers: int,
        records_per_buffer: int = 1,
    ):
        self.sigSettingsUpdated.emit(self._settings.piezo_settings)  # type: ignore

        super()._apply_configuration(
            settings,
            mode,
            num_buffers,
            records_per_buffer,
        )

    # TODO: Camryn: Implement this function. This is what I have for the mIRage
    # system -- not sure if yours should be the same (or not...)
    # It needs to return how many samples you want per record, which for us, is
    # equivalent to how many there are per buffer. For the mIRage, this is
    # one pixel-wavelength (e.g. samples to fill the IR laser dwell time at a
    # given wavelength).
    def _calculate_samples_per_record(self) -> int:
        """Note that this is per-channel"""

        # Each wavelength-pixel gets half the clk freq. of dwell time
        print(f"clk: {self._settings.piezo_settings.clk}")
        print(f"sample_rate: {self._settings.sample_rate}")
        samps = (2 / self._settings.piezo_settings.clk) * self._settings.sample_rate

        if samps < 256:
            self.log.error(
                f"Number of samples per pixel-wavelength is too small: {samps}. Increase pixel dwell time or resolution of image."
            )

        if not samps % 1 < self._min_mod:
            self.log.error(
                f"Samples per pixel-wavelength is ({samps}) which is not within ({self._min_mod}) of an integer. Adjust width resolution or mirror period."
            )

        if not samps % 32 == 0:
            # remainder = samps % 32
            # added = samps + remainder
            # subtracted = samps - remainder
            # guess = added
            # if not guess % 32 == 0:
            #     guess = subtracted

            # guess = 1e6 * guess / self._settings.sample_rate
            self.log.error(
                f"Number of samples per pixel-wavelength is not a multiple of 32, which is required by Alazar. Number of samples was {samps}"
            )

        return int(samps)

    def _calculate_total_samples(self, board_idx: int) -> int:
        return super()._calculate_total_samples(board_idx)

    def _initialize_data(self):
        super()._initialize_data()

    def _update_display_data(self):
        if self._buffer_index % self._settings.num_frames == 0:
            super()._update_display_data()

    # @QtCore.Slot()  # type: ignore
    # def _stage_armed(self):
    # pass

    @QtCore.Slot()  # type: ignore
    def _stage_armed(self):
        kind = self._pending_start_kind
        self._pending_start_kind = None

        if kind == "normal":
            super().start_acquisition()
        elif kind == "live":
            super().start_live_acquisition()
        else:
            # No pending start requested; nothing to do
            return
