# -*- coding: utf-8 -*-

__all__ = ["BaseAlazarLogic"]

from qudi.core.module import LogicBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore
from qudi.util.datastorage import TextDataStorage  # type: ignore

from qudi.interface.alazar_interface import BoardInfo, AlazarInterface, AcquisitionMode

from PySide2 import QtCore
from abc import ABC, abstractmethod
from types import FunctionType
from typing import TypeVar, Generic
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
from importlib import import_module
from enum import Enum, auto

ExperimentSettings = TypeVar("ExperimentSettings", bound="BaseExperimentSettings")


class BaseExperimentSettings(ABC):
    """
    Each type of expriment _must_ implement this class for passing to processing
    functions (live or end). E.g. for GalvoResLogic there should also be a
    GalvoResExperimentSettings that defines relevant information
    """

    autosave_file_path: str | None
    do_autosave: bool
    live_process_function: str | None
    end_process_function: str | None
    sample_rate: int

    @abstractmethod
    def __init__(
        self,
        autosave_file_path: str | None = None,
        do_autosave: bool = False,
        live_process_function: str | None = None,
        end_process_function: str | None = None,
        sample_rate: int = 50_000_000,
    ):
        self.autosave_file_path = autosave_file_path
        self.do_autosave = do_autosave
        self.live_process_function = live_process_function
        self.end_process_function = end_process_function
        self.sample_rate = sample_rate


class ImagingExperimentSettings(BaseExperimentSettings):
    """
    Meta-class that inficates the data will (or can be) imaged in a regular-ish
    way.
    """

    width: int
    height: int
    mirror_period_us: float  # in us
    fast_mirror_phase: float
    num_frames: int

    @abstractmethod
    def __init__(
        self,
        width: int,
        height: int,
        autosave_file_path: str | None = None,
        do_autosave: bool = False,
        live_process_function: str | None = None,
        end_process_function: str | None = None,
        sample_rate: int = 50_000_000,
        mirror_period_us: float = 1000.0,
        fast_mirror_phase: float = 0,
        num_frames: int = 1,
    ):
        self.width = width
        self.height = height
        self.mirror_period_us = mirror_period_us
        self.fast_mirror_phase = 0
        self.num_frames = num_frames

        super().__init__(
            autosave_file_path,
            do_autosave,
            live_process_function,
            end_process_function,
            sample_rate,
        )


class DisplayType(Enum):
    IMAGE = auto()
    LINE = auto()


class DisplayData:
    """
    Class to store data that is meant to be viewed (live or otherwise). Is mostly
    just a convenient way to store a label next to the data it labels.
    """

    type: DisplayType
    label: str
    data: npt.NDArray[np.float_]

    def __init__(
        self,
        data: npt.NDArray[np.float_],
        type: DisplayType = DisplayType.IMAGE,
        label: str = "",
    ):
        self.type = type
        self.label = label
        self.data = data

    def add_data(self, data: npt.NDArray[np.float_]):
        self.data = self.data + data

    def divide_data(self, divisor: float):
        self.data = self.data / divisor


class BaseAlazarLogic(LogicBase, Generic[ExperimentSettings]):
    """
    This contains logic that is common to all Alazar experiments. The intention
    is to separate out as much as possible that is shared between the various
    experiments

    Example config that goes into the config file:

    example_logic:
        module.Class: 'base_alazar_logic.BaseAlazarLogic'
        connect:
            alazar: alazar
        options:
            num_buffers: 0 # 0 to use as many buffers as needed to prevent
                           # overflows. However, this might exhaust
                           # RAM, so can set to a number (>= 2) to use that many
                           # buffers and hope we process them fast enough to not
                           # hit an overrun

            image_at_end: True # if the end_function produces data that is appropriate for imaging
            live_viewing_fn: imaging # Function to use for live-viewing results
    """

    # Config Values:
    _num_buffers: int = ConfigOption(name="num_buffers", default=0, missing="info")  # type: ignore
    _image_at_end: bool = ConfigOption(
        name="image_at_end", default=False, missing="info"
    )  # type: ignore
    _live_viewing_fn: str = ConfigOption(
        name="live_viewing_fn", default="", missing="info"
    )  # type: ignore

    # Status Values:
    _boards: list[BoardInfo] = StatusVar(
        name="boards",
        default=[],
        constructor=lambda yaml: [BoardInfo.constructor_func(i) for i in yaml],  # type: ignore
        representer=lambda data: [BoardInfo.representer_func(i) for i in data],  # type: ignore
    )  # type: ignore

    # Declare signals to send events to other modules connecting to this module
    sigBoardInfo = QtCore.Signal(object)  # is a list[BoardInfo]
    sigAcquisitionStarted = QtCore.Signal()
    sigLiveAcquisitionStarted = QtCore.Signal()
    sigAcquisitionCompleted = QtCore.Signal()
    sigAcquisitionAborted = QtCore.Signal()
    sigProgressUpdated = QtCore.Signal(float)
    sigDataUpdated = QtCore.Signal(object)  # is list[np.ndarray]
    sigImageDataUpdated = QtCore.Signal(object)  # is list[DisplayData]

    # Declare connectors to other logic modules or hardware modules to interact with
    _alazar = Connector(name="alazar", interface="AlazarInterface")

    _settings: ExperimentSettings
    _board_data_index: int = 0
    _buffer_index: int = 0
    _data: list[npt.NDArray[np.float_]] = []
    _display_data: list[DisplayData] = []
    _live_fn: FunctionType | None
    _end_fun = FunctionType | None
    _running_live: bool = False

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    @abstractmethod
    def on_activate(self) -> None:
        alazar: AlazarInterface = self._alazar()
        #### INPUTS #####
        alazar.sigNewData.connect(self._new_alazar_data, QtCore.Qt.QueuedConnection)  # type: ignore
        alazar.sigAcquisitionCompleted.connect(  # type: ignore
            self._acquisition_completed, QtCore.Qt.QueuedConnection
        )

        #### OUTPUTS #####
        self.sigAcquisitionStarted.connect(  # type: ignore
            alazar.start_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigLiveAcquisitionStarted.connect(  # type: ignore
            alazar.start_live_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigAcquisitionAborted.connect(  # type: ignore
            alazar.stop_acquisition,
            QtCore.Qt.DirectConnection,  # needs to be direct
        )

        self.sigBoardInfo.connect(alazar.configure_boards, QtCore.Qt.QueuedConnection)  # type: ignore

        # Get board info:
        boards = alazar.boards_info

        if len(boards) == len(self._boards):
            for i, b in enumerate(boards):
                if len(self._boards[i].channels) == len(b.channels):
                    continue
                else:
                    self._boards[i].channels = b.channels
        else:
            self._boards = boards

        self.sigBoardInfo.emit(self._boards)  # type: ignore
        self._settings.sample_rate = alazar.sample_rate

    @abstractmethod
    def on_deactivate(self) -> None:
        pass

    @property
    @abstractmethod
    def board_info(self) -> list[BoardInfo]:
        return self._boards

    @property
    @abstractmethod
    def experiment_info(self) -> ExperimentSettings:
        return self._settings

    @QtCore.Slot(object)  # type: ignore
    @abstractmethod
    def update_boards(self, boards: list[BoardInfo]):
        self._boards = boards
        self.sigBoardInfo.emit(self._boards)  # type: ignore

    @QtCore.Slot(np.ndarray)  # type: ignore
    @abstractmethod
    def _new_alazar_data(self, buf: npt.NDArray[np.int_]):
        board_idx = self._board_data_index % len(self._boards)

        if self._live_fn is not None:
            self._data = self._live_fn(
                self._data,
                buf,
                self._settings,
                self._buffer_index,
                board_idx,
                self._boards,
            )

        else:
            # TODO: check if we need to copy here
            start_idx = (
                self._buffer_index
                * self._calculate_samples_per_record()
                * self._boards[board_idx].count_enabled()
            )
            end_idx = (
                start_idx
                + self._calculate_samples_per_record()
                * self._boards[board_idx].count_enabled()
            )

            self._data[board_idx][start_idx:end_idx] = buf[:]

        self._board_data_index += 1

        if board_idx == len(self._boards) - 1:
            self._buffer_index += 1
            if self._running_live:
                # This assumes that the data is in a shape that is appropriate
                # for imaging
                self._update_display_data()

            self.sigDataUpdated.emit(self._data)  # type: ignore

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def _acquisition_completed(self):
        self.sigAcquisitionCompleted.emit()  # type: ignore
        self.sigProgressUpdated.emit(100.0)  # type: ignore

        if self._end_fn is not None:
            self._data = self._end_fn(
                self._data,
                self._settings,
                self._boards,
            )

        if self._settings.do_autosave:
            self.save_data()

        if self._image_at_end:
            self._update_display_data()
            self.sigImageDataUpdated.emit(self._display_data)  # type: ignore

        self._running_live = False

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def start_acquisition(self):
        """
        This clears data before starting, so if you want to initialize
        arrays, do it after calling super().start_acquisition()
        """
        self._check_config()
        self._data = []
        self._display_data = []
        self._board_data_index = 0
        self._buffer_index = 0
        self.sigAcquisitionStarted.emit()  # type: ignore

        if self._live_fn is None:
            self.log.info(
                "No live processing function specified, will try to store all data until the end of the acquisition"
            )
            self._initialize_data()

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def start_live_acquisition(self):
        """
        This clears data before starting, so if you want to initialize
        arrays, do it after calling super().start_live_acquisition()
        """
        self._check_config()
        self._data = []
        self._board_data_index = 0
        self._buffer_index = 0
        self._running_live = True
        self.sigLiveAcquisitionStarted.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def stop_acquisition(self):
        self._running_live = False
        self.sigAcquisitionAborted.emit()  # type: ignore

    @QtCore.Slot(BaseExperimentSettings)  # type: ignore
    @abstractmethod
    def configure_acquisition(
        self,
        settings: ExperimentSettings,
    ):
        self._settings = settings

    @abstractmethod
    def _calculate_samples_per_record(self) -> int:
        pass

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def save_data(self):
        path = Path(
            self._settings.autosave_file_path
            if self._settings.autosave_file_path is not None
            else ""
        )
        os.makedirs(path.parent)

        storage = TextDataStorage(
            root_dir=path.parent,
            comments="# ",
            delimiter="\t",
            file_extension=".dat",
            include_global_metadata=False,
        )

        storage.save_data(self._data, filename=path.absolute())  # type: ignore

    @abstractmethod
    def _check_config(self):
        for i, b in enumerate(self._boards):
            if not b.valid_conf():
                self.log.error(
                    f"Not a valid configuration for board {i}. Need at least one channel and a power of 2 total channels."
                )
                raise ValueError("Invalid channel configuration")

    @abstractmethod
    def _apply_configuration(
        self,
        settings: ExperimentSettings,
        mode: AcquisitionMode,
        num_buffers: int,
        records_per_buffer: int = 1,
    ):
        self._num_buffers = num_buffers
        alazar: AlazarInterface = self._alazar()

        alazar.set_samples_per_record(self._calculate_samples_per_record())
        alazar.set_records_per_buffer(records_per_buffer)
        alazar.set_records_per_acquisition(num_buffers)
        alazar.set_acqusition_flag(mode)
        alazar.set_num_buffers(num_buffers)

        # Load modules for processing functions if they exist:
        if (
            settings.live_process_function is not None
            and len(settings.live_process_function) > 0
        ):
            try:
                mod = import_module(
                    f"processing_functions.live.{settings.live_process_function}"
                )
                self._live_fn = getattr(mod, settings.live_process_function)
            except ImportError as e:
                print(f"Failed to load module {settings.live_process_function}: {e}")
            except AttributeError as e:
                print(
                    f"Function '{settings.live_process_function}' not found in module: {e}"
                )

            if self._live_fn is None:
                self.log.error(
                    "Could not find live function, make sure it is declared in './processing_functions/live in a file with the same name as the function"
                )
        else:
            self._live_fn = None

        if (
            settings.end_process_function is not None
            and len(settings.end_process_function) > 0
        ):
            try:
                mod = import_module(
                    f"processing_functions.end.{settings.end_process_function}"
                )
                self._end_fn = getattr(mod, settings.end_process_function)
            except ImportError as e:
                print(f"Failed to load module {settings.end_process_function}: {e}")
            except AttributeError as e:
                print(
                    f"Function '{settings.end_process_function}' not found in module: {e}"
                )

            if self._end_fn is None:
                self.log.error(
                    "Could not find end function, make sure it is declared in './processing_functions/end in a file with the same name as the function"
                )
        else:
            self._end_fn = None

    @abstractmethod
    def _calculate_total_samples(self, board_idx: int) -> int:
        return (
            self._calculate_samples_per_record()
            * self._num_buffers
            * self._boards[board_idx].count_enabled()
        )

    @abstractmethod
    def _initialize_data(self):
        self._data.clear()
        for i in range(len(self._boards)):
            self._data.append(np.zeros(self._calculate_total_samples(i)))

    @abstractmethod
    def _update_display_data(self):
        """
        Copies data from _data into _display data, and applies a basic label to it
        """
        self._display_data.clear()
        for i in range(len(self._data)):
            d = self._data[i]
            t: DisplayType | None = None
            if d.ndim == 1:
                t = DisplayType.LINE
            if d.ndim == 2:
                t = DisplayType.IMAGE

            if t is None:
                self.log.error(
                    f"Data has wrong dimenions for imaging: {d.ndim}. It should have 1 or 2 dimensions."
                )
            else:
                self._display_data.append(
                    DisplayData(type=t, label=f"Data {i}", data=d)
                )

        self.sigImageDataUpdated.emit(self._display_data)  # type: ignore
