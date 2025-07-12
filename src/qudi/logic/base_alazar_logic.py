# -*- coding: utf-8 -*-

__all__ = ["BaseAlazarLogic"]

from PySide2 import QtCore
from abc import ABC, abstractmethod
from types import FunctionType
from typing import TypeVar, Generic
import numpy as np
import os
from pathlib import Path
from importlib import import_module

from qudi.core.module import LogicBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore
from qudi.util.datastorage import TextDataStorage  # type: ignore

from qudi.interface.alazar_interface import BoardInfo, AlazarInterface, AcquisitionMode

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

    @abstractmethod
    def __init__(
        self,
        autosave_file_path: str | None = None,
        do_autosave: bool = False,
        live_process_function: str | None = None,
        end_process_function: str | None = None,
    ):
        self.autosave_file_path = autosave_file_path
        self.do_autosave = do_autosave
        self.live_process_function = live_process_function
        self.end_process_function = end_process_function


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
            live_viewing_fn: template_live_view # Function to use for live-viewing results
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
        constructor=lambda yaml: [BoardInfo.constructor_func(i) for i in yaml],
        representer=lambda data: [BoardInfo.representer_func(i) for i in data],
    )  # type: ignore

    # Declare signals to send events to other modules connecting to this module
    sigBoardInfo = QtCore.Signal(object)  # is a list[BoardInfo]
    sigAcquisitionStarted = QtCore.Signal()
    sigAcquisitionCompleted = QtCore.Signal()
    sigAcquisitionAborted = QtCore.Signal()
    sigProgressUpdated = QtCore.Signal(float)
    sigDataUpdated = QtCore.Signal(object)  # is list[np.ndarray]
    sigImageDataUpdated = QtCore.Signal(object)  # is list[np.ndarray]

    # Declare connectors to other logic modules or hardware modules to interact with
    _alazar = Connector(name="alazar", interface="AlazarInterface")

    _settings: ExperimentSettings
    _board_data_index: int = 0
    _buffer_index: int = 0
    _data: list[np.ndarray] = []
    _live_fn: FunctionType | None
    _end_fun = FunctionType | None
    _running_live: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def on_activate(self) -> None:
        alazar: AlazarInterface = self._alazar()
        #### INPUTS #####
        alazar.sigNewData.connect(self._new_alazar_data, QtCore.Qt.QueuedConnection)
        alazar.sigAcquisitionCompleted.connect(
            self._acquisition_completed, QtCore.Qt.QueuedConnection
        )

        #### OUTPUTS #####
        self.sigAcquisitionStarted.connect(
            alazar.start_acquisition, QtCore.Qt.QueuedConnection
        )

        self.sigAcquisitionAborted.connect(
            alazar.stop_acquisition,
            QtCore.Qt.DirectConnection,  # needs to be direct
        )

        self.sigBoardInfo.connect(alazar.configure_boards, QtCore.Qt.QueuedConnection)

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

        self.sigBoardInfo.emit(self._boards)
        self._sample_rate = alazar.sample_rate

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

    @QtCore.Slot(object)
    @abstractmethod
    def update_boards(self, boards: list[BoardInfo]):
        self._boards = boards
        self.sigBoardInfo.emit(self._boards)

    @QtCore.Slot(np.ndarray)
    @abstractmethod
    def _new_alazar_data(self, buf: np.ndarray):
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
            self.sigDataUpdated.emit(self._data)
            if self._running_live:
                # This assumes that the data is in a shape that is appropriate
                # for imaging
                self.sigImageDataUpdated.emit(self._data)

    @QtCore.Slot()
    @abstractmethod
    def _acquisition_completed(self):
        self.sigAcquisitionCompleted.emit()
        self.sigProgressUpdated.emit(100.0)

        if self._end_fn is not None:
            self._data = self._end_fn(
                self._data,
                self._settings,
                self._boards,
            )

        if self._settings.do_autosave:
            self.save_data()

        if self._image_at_end:
            self.sigImageDataUpdated.emit(self._data)

        self._running_live = False

    @QtCore.Slot()
    @abstractmethod
    def start_acquisition(self):
        """
        This clears data before starting, so if you want to initialize
        arrays, do it after calling super().start_acquisition()
        """
        self._check_config()
        self._data = []
        self._board_data_index = 0
        self._buffer_index = 0
        self.sigAcquisitionStarted.emit()

        if self._live_fn is None:
            self.log.info(
                "No live processing function specified, will try to store all data until the end of the acquisition"
            )
            self._initialize_data()

    @QtCore.Slot()
    @abstractmethod
    def start_live_acquisition(self, buffers_to_avg: int):
        """
        This clears data before starting, so if you want to initialize
        arrays, do it after calling super().start_live_acquisition()
        """

        self._check_config()
        self._data = []
        self._board_data_index = 0
        self._buffer_index = 0
        self._running_live = True
        self.sigAcquisitionStarted.emit()

    @QtCore.Slot()
    @abstractmethod
    def stop_acquisition(self):
        self._running_live = False
        self.sigAcquisitionAborted.emit()

    @QtCore.Slot(BaseExperimentSettings)
    @abstractmethod
    def configure_acquisition(
        self,
        settings: ExperimentSettings,
    ):
        self._settings = settings

    @abstractmethod
    def _calculate_samples_per_record(self) -> int:
        pass

    @QtCore.Slot()
    @abstractmethod
    def save_data(self):
        path = Path(self._settings.autosave_file_path)
        os.makedirs(path.parent)

        storage = TextDataStorage(
            root_dir=path.parent,
            comments="# ",
            delimiter="\t",
            file_extension=".dat",
            include_global_metadata=False,
        )

        storage.save_data(self._data, filename=path.absolute())

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

    # @_boards.constructor
    # @staticmethod
    # def _boards_constructor(_, yaml_data):
    #     if len(yaml_data) > 0:
    #         return BoardInfo.constructor_func(**yaml_data)

    # @_boards.representer
    # @staticmethod
    # def _boards_representer(_, boards: list[BoardInfo]) -> list[dict[str, object]]:
    #     out: list[dict[str, object]] = []
    #     for b in boards:
    #         out.append(BoardInfo.representer_func(b))

    #     return out
