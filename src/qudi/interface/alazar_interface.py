# -*- coding: utf-8 -*-
from __future__ import annotations

__all__ = ["AlazarInterface"]

from abc import abstractmethod

from enum import Enum
import numpy as np
from qudi.core.module import Base  # type: ignore
from qudi.core.statusvariable import StatusVar  # type: ignore

from PySide2 import QtCore


class Coupling(Enum):
    AC = 0
    DC = 1


class Termination(Enum):
    OHM_50 = 0
    OHM_1M = 1


class Range(Enum):
    PM_200_MV = 0
    PM_500_MV = 1
    PM_1_V = 2
    PM_2_V = 3
    PM_5_V = 4


class AcquisitionMode(Enum):
    TRIGGERED_STREAMING = 0
    NPT = 1


class MeasurementType(Enum):
    UNUSED = 0
    AVERAGING = 1
    PHOTON_COUNTING = 2


class ChannelInfo:
    coupling: Coupling
    termination: Termination
    range: Range
    enabled: bool = False
    measurement_type: MeasurementType = MeasurementType.UNUSED
    label: str

    def __init__(
        self,
        term: Termination = Termination.OHM_50,
        range: Range = Range.PM_200_MV,
        coupling: Coupling = Coupling.DC,
        enabled: bool = False,
        measurement_type: MeasurementType = MeasurementType.UNUSED,
        label: str = "",
    ):
        self.coupling = coupling
        self.termination = term
        self.range = range
        self.enabled = enabled
        self.measurement_type = measurement_type
        self.label = label

    def __str__(self):
        return f"{self.label} enabled:{self.enabled} {self.range.name} {self.coupling.name} {self.termination.name} {self.measurement_type.name}"

    def __repr__(self):
        return f"<ChannelInfo {self.label} enabled:{self.enabled} {self.range.name} {self.coupling.name} {self.termination.name} {self.measurement_type.name}>"

    @staticmethod
    def representer_func(instance: ChannelInfo) -> list[object]:
        return [
            instance.termination.value,
            instance.range.value,
            instance.coupling.value,
            instance.enabled,
            instance.measurement_type.value,
            instance.label,
        ]

    @staticmethod
    def constructor_func(yaml_data: list[object]) -> ChannelInfo:
        return ChannelInfo(
            term=Termination(yaml_data[0]),
            range=Range(yaml_data[1]),
            coupling=Coupling(yaml_data[2]),
            enabled=yaml_data[3],
            measurement_type=MeasurementType(yaml_data[4]),
            label=yaml_data[5],
        )


class BoardInfo:
    channels: list[ChannelInfo]
    label: str

    def __init__(self, channels: list[ChannelInfo], label: str = ""):
        self.channels = channels
        self.label = label

    def __str__(self):
        channel_str = ""
        for c in self.channels:
            channel_str += f"{c}\n"
        return f"{self.label} with channels:\n{channel_str}"

    def __repr__(self):
        return f"<BoardInfo label:{self.label} channels:{self.channels}>"

    def count_enabled(self) -> int:
        count = 0
        for c in self.channels:
            if c.enabled:
                count += 1

        return count

    def valid_conf(self) -> bool:
        # see: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two/73902024#73902024
        n = self.count_enabled()
        return n > 0 and n.bit_count() == 1

    @staticmethod
    def representer_func(instance: BoardInfo) -> list[object]:
        return [
            [ChannelInfo.representer_func(i) for i in instance.channels],
            instance.label,
        ]

    @staticmethod
    def constructor_func(yaml_data: list[object]) -> BoardInfo:
        label = yaml_data.pop()
        chans = []
        for val in yaml_data[0]:
            chans.append(ChannelInfo.constructor_func(val))
        return BoardInfo(channels=chans, label=label)


class AlazarInterface(Base):
    """Interface for Alazar Card(s)"""

    sigNewData = QtCore.Signal(np.ndarray)
    sigAcquisitionCompleted = QtCore.Signal()

    # StatusVars:
    _samples_per_record = StatusVar(name="samples_per_record", default=-1)
    _records_per_buffer = StatusVar(name="records_per_buffer", default=1)
    _records_per_acquisition = StatusVar(name="records_per_acquisition", default=1)
    _num_buffers = StatusVar(name="num_buffers", default=1)
    _adma_flags = StatusVar(
        name="adma_flags",
        default=0x1000
        + 0x1
        + 0x200,  # ats.ADMA_INTERLEAVE_SAMPLES + ats.ADMA_EXTERNAL_STARTCAPTURE + ats.ADMA_NPT,
    )

    @property
    @abstractmethod
    def boards_info(self) -> list[BoardInfo]:
        """
        Returns a list for how many boards are in the system that contains
        information about how many channels each board has
        """
        pass

    @property
    @abstractmethod
    def running(self) -> bool:
        """
        Returns whether the card is currently acquiring data
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """
        Sample rate in Hz
        """
        pass

    @property
    @abstractmethod
    def samples_per_buffer(self) -> int:
        """
        Samples per buffer
        """
        pass

    @abstractmethod
    def set_samples_per_record(self, samples: int):
        pass

    @abstractmethod
    def set_records_per_buffer(self, records: int):
        pass

    @abstractmethod
    def set_records_per_acquisition(self, records: int):
        pass

    @abstractmethod
    def set_num_buffers(self, num_buffers: int):
        pass

    @QtCore.Slot()
    @abstractmethod
    def start_acquisition(self):
        pass

    @QtCore.Slot()
    @abstractmethod
    def start_live_acquisition(self):
        pass

    @QtCore.Slot()
    @abstractmethod
    def stop_acquisition(self):
        pass

    @abstractmethod
    def set_aux_out(self, high: bool):
        pass

    @QtCore.Slot(object)
    @abstractmethod
    def set_acqusition_flag(self, flag: AcquisitionMode):
        pass

    @QtCore.Slot(object)
    @abstractmethod
    def configure_boards(self, boards: list[BoardInfo]):
        """
        Update settings for the boards / channels. Need to preserve order for
        the boards (originally from boards_info)
        """
        pass
