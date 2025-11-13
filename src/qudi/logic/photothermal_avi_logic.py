from __future__ import annotations

__all__ = ["PhotothermalAVILogic", "PhotothermalAVISettings"]

from PySide2 import QtCore
import numpy as np
import numpy.typing as npt
from logging import Logger
from typing import cast


from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore
from qudi.core.connector import Connector  # type: ignore

from qudi.logic.base_alazar_logic import BaseAlazarLogic
from qudi.logic.experiment_defs import ImagingExperimentSettings
from qudi.interface.alazar_interface import BoardInfo, AcquisitionMode
from qudi.interface.piezo_stage_interface import PiezoScanSettings, PiezoStageInterface


class PhotothermalAVISettings(ImagingExperimentSettings):
    wavelengths_per_pixel: int
    piezo_settings: PiezoScanSettings
    records_per_buffer: int

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
        self.records_per_buffer = 1
        self.adc_midcode = 32768.0
        self.invert_polarity = True
        self.line_head_trim = 0
        self.line_tail_trim = 0
        self.bidirectional = True
        self.transpose_image = False
        self.frame_head_rows_skip = 8
        self.reverse_retrace = False
        self.display_fixed_levels = None

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
        return int(self.height)

    def scan_freq_hz(self) -> float:
        return 1e6 / self.fast_motion_period_us

    def update_piezo_settings(self, logger: Logger) -> PiezoScanSettings:
        """
        Configure galvo timing so that one DMA record equals one full fast period
        (trace plus retrace), with fast axis frequency doubled relative to the
        original 1024 step configuration.
        """
        a_wave_on = self.wavelengths_per_pixel
        a_wave_off = 0
        fast_wave_b_pulses = 1

        # Half the steps to double the line rate.
        out_settings = PiezoScanSettings(
            a_wave_on=a_wave_on,
            a_wave_off=a_wave_off,
            fast_wave_b_pulses=fast_wave_b_pulses,
            fast_wave_ramp_steps=512,   # trace-only period = 512 pulses → 4 second frame
            fast_wave_scans_per_slow=1,
            slow_wave_ramp_steps=self.height,
            slow_wave_scans_per_trigger=1,   # exactly one slow ramp
            slow_wave_enable_mode=self.piezo_settings.slow_wave_enable_mode,
        )

        self.piezo_settings = out_settings
        return out_settings

    @staticmethod
    def representer_func(instance: PhotothermalAVISettings) -> list[object]:
        return [
            instance.fast_motion_phase,
            instance.fast_motion_period_us,
            instance.wavelengths_per_pixel,
            instance.piezo_settings,
            instance.width,
            instance.height,
            instance.num_frames,
            instance.autosave_file_path,
            instance.do_autosave,
            instance.live_process_function,
            instance.end_process_function,
        ]

    @staticmethod
    def constructor_func(yaml_data: object) -> "PhotothermalAVISettings":
        return cast(PhotothermalAVISettings, yaml_data)



class PhotothermalAVILogic(BaseAlazarLogic[PhotothermalAVISettings]):
    _fast_motion_phase: float = ConfigOption(
        name="fast_motion_phase", default=0.0, missing="info"
    )  # type: ignore
    _min_mod: float = ConfigOption(name="min_mod", default=1e-12, missing="info")  # type: ignore
    _display_fixed_levels: tuple[float, float] | None = ConfigOption(
        name="display_fixed_levels", default=None, missing="info"
    )  # type: ignore

    _settings: PhotothermalAVISettings = StatusVar(
        name="photothermal_avi_settings",
        default=PhotothermalAVISettings(),
        constructor=PhotothermalAVISettings.constructor_func,
        representer=PhotothermalAVISettings.representer_func,
    )  # type: ignore

    _galvos = Connector(name="galvos", interface="PiezoStageInterface")

    sigSettingsUpdated = QtCore.Signal(object)
    sigStartStage = QtCore.Signal()
    sigStopStage = QtCore.Signal()
    _pending_start_kind: str | None = None

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        galvos: PiezoStageInterface = self._galvos()

        p_settings = galvos.get_settings
        self._settings.piezo_settings.clk = p_settings.clk
        self._settings.piezo_settings.wave_f_mode = p_settings.wave_f_mode
        self._settings.piezo_settings.enable_polarity = p_settings.enable_polarity
        self._settings.piezo_settings.slow_wave_enable_mode = p_settings.slow_wave_enable_mode

        self.sigSettingsUpdated.connect(galvos.update_settings, QtCore.Qt.QueuedConnection)  # type: ignore
        self.sigStartStage.connect(galvos.download_and_arm, QtCore.Qt.QueuedConnection)  # type: ignore
        self.sigStopStage.connect(galvos.end_scan, QtCore.Qt.QueuedConnection)  # type: ignore

        galvos.sigStageArmed.connect(self._stage_armed)  # type: ignore

        self._settings.fast_motion_phase = self._fast_motion_phase
        self._settings.display_fixed_levels = self._display_fixed_levels  # type: ignore[attr-defined]

        super().on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()

    @property
    def board_info(self) -> list[BoardInfo]:
        return super().board_info

    @property
    def experiment_info(self) -> PhotothermalAVISettings:
        return super().experiment_info

    @QtCore.Slot(object)  # type: ignore
    def update_boards(self, boards: list[BoardInfo]):
        super().update_boards(boards)

    @QtCore.Slot(np.ndarray)  # type: ignore
    def _new_alazar_data(self, buf: npt.NDArray[np.int_]):
        super()._new_alazar_data(buf)

    @QtCore.Slot()  # type: ignore
    def _acquisition_completed(self):
        if not self._running_live:
            self.sigStopStage.emit()  # type: ignore
        super()._acquisition_completed()

    @QtCore.Slot()  # type: ignore
    def start_acquisition(self):
        self._settings.live_process_function = self._live_viewing_fn
        self._settings.end_process_function = None
        self._num_buffers = 0
        num_b = self._settings.calc_records_per_acquisition()

        print(
            f"[PLAN] normal width={self._settings.width} height={self._settings.height} "
            f"records_per_buffer={self._settings.records_per_buffer} "
            f"records_per_acq={self._settings.calc_records_per_acquisition()} "
            f"num_buffers_planned={num_b}"
        )

        self._apply_configuration(
            settings=self._settings,
            mode=AcquisitionMode.TRIGGERED_STREAMING,
            num_buffers=num_b,
            records_per_buffer=self._settings.records_per_buffer,
        )
        self._pending_start_kind = "normal"
        super().start_acquisition()

    @QtCore.Slot(int)  # type: ignore
    def start_live_acquisition(self):
        live_settings = self._settings
        live_settings.num_frames = self._settings.num_frames
        live_settings.live_process_function = "photothermal_avi"
        live_settings.bidirectional = False
        live_settings.transpose_image = True
        live_settings.piezo_settings.slow_wave_scans_per_trigger = 32767

        live_mode = AcquisitionMode.TRIGGERED_STREAMING

        frame_buffers = int(live_settings.height)
        buffers_for_live = frame_buffers

        self._apply_configuration(
            settings=live_settings,
            mode=live_mode,
            num_buffers=buffers_for_live,
            records_per_buffer=1,
        )

        self._alazar().set_records_per_acquisition(10_000_000)  # type: ignore

        self._pending_start_kind = "live"
        super().start_live_acquisition()

    @QtCore.Slot()  # type: ignore
    def stop_acquisition(self):
        self.sigStopStage.emit()  # type: ignore
        super().stop_acquisition()

    @QtCore.Slot(object)  # type: ignore
    def configure_acquisition(self, settings: PhotothermalAVISettings):
        p_settings = self._settings.piezo_settings
        super().configure_acquisition(settings)
        self._settings.piezo_settings = p_settings
        self._settings.fast_motion_phase = self._fast_motion_phase
        self._settings.records_per_buffer = 1
        self._settings.update_piezo_settings(logger=self.log)

        fast_steps = int(
            getattr(self._settings.piezo_settings, "fast_wave_ramp_steps", self._settings.width)
        )
        if fast_steps != int(self._settings.width):
            self.log.warning(
                f"Using fast_wave_ramp_steps={fast_steps} with GUI width={self._settings.width}. "
                "Controller timing follows external clock semantics."
            )

        samps = self._calculate_samples_per_record()
        expected = 2 * fast_steps
        if samps != expected:
            self.log.error(
                f"Inconsistent line configuration: samples_per_record={samps} but expected {expected}."
            )
            raise ValueError("Line timing mismatch: samples_per_record must equal 2 * fast_steps.")


        if samps % 32 != 0:
            self.log.error(
                f"samples_per_record={samps} is not a multiple of 32."
            )
            raise ValueError("samples_per_record must be a multiple of 32.")

    @QtCore.Slot()  # type: ignore
    def save_data(self):
        super().save_data()

    @QtCore.Slot()  # type: ignore
    def _board_armed(self):
        self.sigStartStage.emit()  # type: ignore

    def _check_config(self):
        super()._check_config()

    def _apply_configuration(
        self,
        settings: PhotothermalAVISettings,
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

    def _calculate_samples_per_record(self) -> int:
        """
        AVI hardware outputs a full triangle: trace + retrace.
        Total samples per record = 2 * fast_steps = 1024.
        """
        steps = int(getattr(self._settings.piezo_settings, "fast_wave_ramp_steps"))
        return 2 * steps






    def _calculate_total_samples(self, board_idx: int) -> int:
        return super()._calculate_total_samples(board_idx)

    def _initialize_data(self):
        super()._initialize_data()

    def _update_display_data(self):
        if self._buffer_index % self._settings.num_frames == 0:
            super()._update_display_data()

    @QtCore.Slot()  # type: ignore
    def _stage_armed(self):
        kind = self._pending_start_kind or "unknown"
        self.log.info(
            f"Galvo stage armed; pending_start_kind={kind}. Waiting for SCAN ACTIVE to trigger capture."
        )
        self._pending_start_kind = None
