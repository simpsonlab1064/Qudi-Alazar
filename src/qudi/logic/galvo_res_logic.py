from __future__ import annotations

__all__ = ["GalvoResLogic"]

from PySide2 import QtCore
import numpy as np
import numpy.typing as npt

from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore
from qudi.logic.base_alazar_logic import (
    BaseAlazarLogic,
    ImagingExperimentSettings,
)
from qudi.interface.alazar_interface import BoardInfo, AcquisitionMode


# Note: do_series is currently unused as I don't know what it's actually for.
# Adding it in should be pretty doable once I know what purpose it serves
class GalvoResExperimentSettings(ImagingExperimentSettings):
    series_length: int
    do_series: bool
    live_process_function: str | None
    end_process_function: str | None

    def __init__(
        self,
        fast_mirror_phase: float = 0,
        mirror_period_us: float = 66.8,
        width: int = 512,
        height: int = 512,
        num_frames: int = 1,
        series_length: int = 1,
        do_series: bool = False,
        autosave_file_path: str | None = None,
        do_autosave: bool = False,
        live_processing_function: str | None = None,
        end_processing_function: str | None = None,
    ):
        self.width = width
        self.height = height
        self.series_length = series_length
        self.do_series = do_series

        super().__init__(
            width=width,
            height=height,
            autosave_file_path=autosave_file_path,
            do_autosave=do_autosave,
            live_process_function=live_processing_function,
            end_process_function=end_processing_function,
            fast_mirror_phase=fast_mirror_phase,
            mirror_period_us=mirror_period_us,
            num_frames=num_frames,
        )

    def scan_freq_hz(self) -> float:
        return 1e6 / self.mirror_period_us

    @staticmethod
    def representer_func(instance: GalvoResExperimentSettings) -> list[object]:
        return [
            instance.fast_mirror_phase,
            instance.mirror_period_us,
            instance.width,
            instance.height,
            instance.num_frames,
            instance.series_length,
            instance.do_series,
            instance.autosave_file_path,
            instance.do_autosave,
            instance.live_process_function,
            instance.end_process_function,
        ]

    @staticmethod
    def constructor_func(yaml_data: object) -> GalvoResExperimentSettings:
        return GalvoResExperimentSettings(*yaml_data)  # type: ignore


class GalvoResLogic(BaseAlazarLogic[GalvoResExperimentSettings]):
    """
    This contains logic for running a galvo-res experiment

    Example config that goes into the config file:

    example_logic:
        module.Class: 'galvo_res_logic.GalvoResLogic'
        connect:
            alazar: alazar
        options:
            num_buffers: 0 # 0 to use as many buffers as needed to prevent
                           # overflows. However, this might exhaust
                           # RAM, so can set to a number (>= 2) to use that many
                           # buffers and hope we process them fast enough to not
                           # hit an overrun
            min_mod: 1e-12 # Maximum value that samples per line can be different
                           # without triggering an error

            image_at_end: True # if the end_function produces data that is appropriate for imaging
    """

    _settings: GalvoResExperimentSettings = StatusVar(
        name="galvo_res_settings",
        default=GalvoResExperimentSettings(),
        constructor=GalvoResExperimentSettings.constructor_func,
        representer=GalvoResExperimentSettings.representer_func,
    )  # type: ignore

    _min_mod: float = ConfigOption(name="min_mod", default=1e-12, missing="info")  # type: ignore

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        super().on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()

    @property
    def board_info(self) -> list[BoardInfo]:
        return super().board_info

    @property
    def experiment_info(self) -> GalvoResExperimentSettings:
        return super().experiment_info

    @QtCore.Slot(object)  # type: ignore
    def update_boards(self, boards: list[BoardInfo]):
        self._boards = boards
        self.sigBoardInfo.emit(self._boards)  # type: ignore

    @QtCore.Slot(np.ndarray)  # type: ignore
    def _new_alazar_data(self, buf: npt.NDArray[np.int_]):
        super()._new_alazar_data(buf)

    @QtCore.Slot()  # type: ignore
    def _acquisition_completed(self):
        super()._acquisition_completed()

    @QtCore.Slot()  # type: ignore
    def start_acquisition(self):
        self._apply_configuration(
            settings=self._settings,
            mode=AcquisitionMode.TRIGGERED_STREAMING,
            num_buffers=self._settings.num_frames,
        )
        super().start_acquisition()

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
        super().start_live_acquisition()

    @QtCore.Slot()  # type: ignore
    def stop_acquisition(self):
        super().stop_acquisition()

    @QtCore.Slot(object)  # type: ignore
    def configure_acquisition(self, settings: GalvoResExperimentSettings):
        super().configure_acquisition(settings)

    @QtCore.Slot()  # type: ignore
    def save_data(self):
        super().save_data()

    def _check_config(self):
        super()._check_config()

    def _apply_configuration(
        self,
        settings: GalvoResExperimentSettings,
        mode: AcquisitionMode,
        num_buffers: int,
        records_per_buffer: int = 1,
    ):
        super()._apply_configuration(
            settings,
            mode,
            num_buffers,
            records_per_buffer,
        )

    def _calculate_samples_per_record(self) -> int:
        """Note that this is per-channel"""
        samps: float = 0
        samples_per_line = (
            1e3
            * self._settings.mirror_period_us
            / (1e9 / float(self._settings.sample_rate))
        )

        samps = samples_per_line * float(self._settings.height)
        self.log.info(f"Samples per line: {samples_per_line}")

        if samps < 256:
            self.log.error(
                f"Number of samples per frame is too small: {samps}. Increase pixel dwell time or resolution of image."
            )

        if not samps % 1 < self._min_mod:
            self.log.error(
                f"Samples for a single line is within allowed tolerance ({self._min_mod}) of an integer. Adjust width resolution or mirror period."
            )

        return int(samps)

    def _calculate_total_samples(self, board_idx: int) -> int:
        return super()._calculate_total_samples(board_idx)

    def _initialize_data(self):
        super()._initialize_data()

    def _update_display_data(self):
        if self._buffer_index % self._settings.num_frames == 0:
            super()._update_display_data()
