from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
import numpy.typing as npt
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
    """In Hz"""

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
