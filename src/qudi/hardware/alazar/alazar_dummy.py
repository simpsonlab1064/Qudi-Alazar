# -*- coding: utf-8 -*-

__all__ = ["AlazarDummy"]

import numpy as np
import time
from PySide2 import QtCore
from qudi.util.mutex import RecursiveMutex  # type: ignore

from qudi.interface.alazar_interface import (
    AlazarInterface,
    ChannelInfo,
    BoardInfo,
    AcquisitionMode,
)


class AlazarDummy(AlazarInterface):
    """Dummy interface for Alazar Card(s)

    Example config for copy-paste:

    alazar_dummy:
        module.Class: 'alazar.alazar_dummy.AlazarDummy'

    """

    _threaded = True
    _sample_rate = 50_000_000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()

        self._boards = self.boards_info

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    @property
    def boards_info(self) -> list[BoardInfo]:
        """
        Returns a list for how many boards are in the system that contains
        information about how many channels each board has
        """
        return [
            BoardInfo(
                [
                    ChannelInfo(label="Chan A 1"),
                    ChannelInfo(label="Chan B 1"),
                ],
                "Board 1",
            ),
            BoardInfo(
                [
                    ChannelInfo(label="Chan A 2"),
                    ChannelInfo(label="Chan B 2"),
                ],
                "Board 2",
            ),
        ]

    @property
    def running(self) -> bool:
        """
        Returns whether the card is currently acquiring data
        """
        return self.module_state() == "locked"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def samples_per_buffer(self) -> int:
        return self._samples_per_record * self._records_per_buffer

    def set_samples_per_record(self, samples: int):
        with self._thread_lock:
            if self.module_state() == "idle":
                self._samples_per_record = samples

    def set_records_per_buffer(self, records: int):
        with self._thread_lock:
            if self.module_state() == "idle":
                self._records_per_buffer = records

    def set_records_per_acquisition(self, records: int):
        with self._thread_lock:
            if self.module_state() == "idle":
                self._records_per_acquisition = records

    def set_num_buffers(self, num_buffers: int):
        with self._thread_lock:
            if self.module_state() == "idle":
                if num_buffers > 0:
                    self._num_buffers = num_buffers
                else:
                    self._num_buffers = (
                        self._records_per_acquisition * self._records_per_buffer
                    )

    @QtCore.Slot()
    def start_acquisition(self):
        if self.module_state() == "idle":
            self.module_state.lock()
            self._acquire_data()

            self.sigAcquisitionCompleted.emit()
            self.module_state.unlock()

    @QtCore.Slot()
    def start_live_acquisition(self):
        if self.module_state() == "idle":
            self.module_state.lock()
            self._acquire_live_data()

            self.sigAcquisitionCompleted.emit()

    @QtCore.Slot()
    def stop_acquisition(self):
        # with self._thread_lock:  # maybe we don't want to acquire the lock here...
        if self.module_state() == "locked":
            self.module_state.unlock()

    def set_aux_out(self, high: bool):
        pass

    @QtCore.Slot(object)
    def set_acqusition_flag(self, flag: AcquisitionMode):
        with self._thread_lock:
            if self.module_state() == "idle":
                self._adma_flags = 0x1

    @QtCore.Slot(object)
    def configure_boards(self, boards: list[BoardInfo]):
        with self._thread_lock:
            if self.module_state() == "idle":
                for i in range(len(boards)):
                    self._boards[i] = boards[i]

    def _acquire_data(self):
        start = time.time()

        self.set_aux_out(True)

        i = 0
        pause = float(self._samples_per_record * self._records_per_buffer) / float(
            self._sample_rate
        )

        while i < self._num_buffers and self.module_state() == "locked":
            for j in range(len(self._boards)):
                buf = self._generate_data(
                    (
                        self._samples_per_record
                        * self._records_per_buffer
                        * self._boards[j].count_enabled()
                    ),
                    j,
                )

                # TODO: check if this needs a .copy() (or not)
                # Maybe do the copy on the other end
                self.sigNewData.emit(buf)

                time.sleep(pause)

            i += 1

        self.set_aux_out(False)

        transfer_time = time.time() - start
        self.log.info(f"Dummy data collection finished in: {transfer_time}")

    def _acquire_live_data(self):
        while self.module_state() == "locked":
            i = 0
            pause = float(self._samples_per_record * self._records_per_buffer) / float(
                self._sample_rate
            )

            while i < self._num_buffers:
                for j in range(len(self._boards)):
                    buf = self._generate_data(
                        (
                            self._samples_per_record
                            * self._records_per_buffer
                            * self._boards[j].count_enabled()
                        ),
                        j,
                    )

                    self.sigNewData.emit(buf)

                    time.sleep(pause)

                i += 1

    def _generate_data(self, num_samples: int, board_idx: int) -> np.ndarray:
        buf = np.random.rand(num_samples)
        # Bright stripe in the middle of the data
        start = round(num_samples / 4)
        end = round(3 * num_samples / 4)

        buf[start:end] = buf[start:end] + 5 * (board_idx + 1)

        return buf
