# -*- coding: utf-8 -*-

__all__ = ["AlazarCard"]

import ctypes
import time
from PySide2 import QtCore
from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.util.mutex import RecursiveMutex  # type: ignore

import qudi.hardware.alazar.Library.atsapi as ats
from qudi.interface.alazar_interface import (
    AlazarInterface,
    ChannelInfo,
    BoardInfo,
    Coupling,
    Range,
    Termination,
    AcquisitionMode,
)


class CombinedBoard:
    info: BoardInfo
    internal: ats.Board

    def __init__(self, info: BoardInfo, internal: ats.Board):
        self.info = info
        self.internal = internal

    def valid_conf(self) -> bool:
        enabled = self.info.count_enabled()
        return enabled > 0 and (enabled % 2 == 0 or enabled == 1)


class AlazarCard(AlazarInterface):
    """
    Interface for reading data from Alazar DAQ cards for experiments in the
    Simpson lab. For other experiments, you likely need to update the call
    to setCaptureClock with appropriate settings and the trigger settings

    Example config for copy-paste:

    alazar:
        module.Class: 'alazar.alazar_card.AlazarCard'
        options:
            systemId: 1 # only if there are multiple systems (not just multiple cards)
            clock: 1 # 0 for internal PLL, 1 for external, 2 for slow external
            sample_rate: 50_000_000 # Sample rate for acquisition, in Hz
            card_type: "c9440" # options are "c9440" or "c9350"
            trigger_level: 160 # 0-255, 0-127 = negative range 128-255 = positive range
            trigger_timeout: 5 # how long to wait (in seconds) for trigger before aborting, set to 0 for infinite
    """

    # Declare static parameters that can/must be declared in the qudi configuration
    _trigger: int = ConfigOption(name="trigger", default=1, missing="warn")  # type: ignore

    _clock: int = ConfigOption(name="clock", default=1, missing="warn")  # type: ignore

    _sample_rate: int = ConfigOption(name="sample_rate", missing="error")  # type: ignore

    _systemId: int = ConfigOption(name="systemId", default=1, missing="info")  # type: ignore

    _card_type: str = ConfigOption(name="card_type", default="c9440", missing="warn")  # type: ignore

    _trigger_level: int = ConfigOption(
        name="trigger_level", default=160, missing="warn"
    )  # type: ignore

    _trigger_timeout: float = ConfigOption(
        name="trigger_timeout", default=5, missing="info"
    )  # type: ignore

    # run in separate thread
    _threaded = True

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

        self._thread_lock = RecursiveMutex()

        self._boards: list[CombinedBoard] = []

    def on_activate(self) -> None:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._buffers: list[list[ats.DMABuffer]] = []
        for i in range(ats.boardsInSystemBySystemID(self._systemId)):  # type: ignore
            b = ats.Board(self._systemId, i + 1)
            num_channels = b.getParameter(  # type: ignore
                parameter=ats.GET_CHANNELS_PER_BOARD, channel=0
            )
            chans = [
                ChannelInfo(label=f"Channel {letters[i]}")
                for i in range(num_channels.value)  # type: ignore
            ]
            info = BoardInfo(channels=chans, label=f"Board {i}")
            self._boards.append(
                CombinedBoard(
                    info=info,
                    internal=b,
                )
            )

            self._buffers.append([])

            self._decimation = 0
            if self._card_type == "c9350":
                self._decimation = 1

            self._sample_type = ctypes.c_uint16

    def on_deactivate(self) -> None:
        for b in self._boards:
            b.internal.abortAsyncRead()

    @property
    def boards_info(self) -> list[BoardInfo]:
        """
        Returns a list for how many boards are in the system that contains
        information about how many channels each board has
        """
        return [i.info for i in self._boards]

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

    @QtCore.Slot()  # type: ignore
    def start_acquisition(self):
        if all([x.valid_conf() for x in self._boards]):
            with self._thread_lock:
                if self.module_state() == "idle":
                    self.module_state.lock()
                    self._configure_and_allocate()

                    if self.module_state() == "locked":
                        self._acquire_data()
                        self.sigAcquisitionCompleted.emit()  # type: ignore

                        self.module_state.unlock()

        else:
            self.log.warning(
                "Not all boards have allowed channels active (at least one channel and if more than one it must be a multiple of 2)"
            )

    @QtCore.Slot()  # type: ignore
    def start_live_acquisition(self):
        if all([x.valid_conf() for x in self._boards]):
            with self._thread_lock:
                if self.module_state() == "idle":
                    self.module_state.lock()
                    self._configure_and_allocate()

                    if self.module_state() == "locked":
                        self._acquire_live_data()
                        self.sigAcquisitionCompleted.emit()  # type: ignore

                        self.module_state.unlock()

        else:
            self.log.warning(
                "Not all boards have allowed channels active (at least one channel and if more than one it must be a multiple of 2)"
            )

    @QtCore.Slot()  # type: ignore
    def stop_acquisition(self):
        # with self._thread_lock:  # maybe we don't want to acquire the lock here...
        if self.module_state() == "locked":
            self.module_state.unlock()

    def set_aux_out(self, high: bool):
        self._boards[0].internal.configureAuxIO(  # type: ignore
            mode=ats.AUX_OUT_SERIAL_DATA,
            parameter=1 if high else 0,
        )

    @QtCore.Slot(object)  # type: ignore
    def set_acqusition_flag(self, flag: AcquisitionMode):
        with self._thread_lock:
            if self.module_state() == "idle":
                self._adma_flags = (
                    ats.ADMA_INTERLEAVE_SAMPLES + ats.ADMA_EXTERNAL_STARTCAPTURE
                )
                if flag == AcquisitionMode.NPT:
                    self._adma_flags += ats.ADMA_NPT

                if flag == AcquisitionMode.TRIGGERED_STREAMING:
                    self._adma_flags += ats.ADMA_TRIGGERED_STREAMING

    @QtCore.Slot(object)  # type: ignore
    def configure_boards(self, boards: list[BoardInfo]):
        with self._thread_lock:
            if self.module_state() == "idle":
                for i in range(len(boards)):
                    self._boards[i].info = boards[i]

    def _configure_and_allocate(self):
        """Expects mutex to be locked externally"""
        if self.module_state() == "locked":
            i = 0
            for b in self._boards:
                self._configure_board(b)
                if self.module_state() == "locked":
                    if self._boards[0].internal.boardId != 1:
                        raise ValueError("The first board passed should be the master.")
                    if b.internal.systemId != self._boards[0].internal.systemId:
                        raise ValueError("All the boards should be of the same system.")
                    if len(self._buffers) > i:
                        self._buffers[i].clear()
                    self._allocate_buffers(b, i)
                    i += 1

    def _configure_board(self, board: CombinedBoard):
        clk = -1
        if self._clock == 0:
            clk = ats.EXTERNAL_CLOCK_10MHz_REF

        if self._clock == 1:
            clk = ats.FAST_EXTERNAL_CLOCK

        if self._clock == 2:
            clk = ats.SLOW_EXTERNAL_CLOCK

        if clk == -1:
            raise ValueError("Clock not set correctly!")
         
        board.internal.setCaptureClock(  # type: ignore
            source=clk,
            rate=self._sample_rate,
            edge=ats.CLOCK_EDGE_RISING,
            decimation=self._decimation,
        )

        numChannels = len(board.info.channels)

        for i in range(numChannels):
            if board.info.channels[i].enabled:
                chan = board.info.channels[i]
                coupling = (
                    ats.DC_COUPLING if chan.coupling == Coupling.DC else ats.AC_COUPLING
                )

                r = -1

                if chan.range == Range.PM_200_MV:
                    r = ats.INPUT_RANGE_PM_200_MV
                elif chan.range == Range.PM_500_MV:
                    r = ats.INPUT_RANGE_PM_500_MV

                elif chan.range == Range.PM_1_V:
                    r = ats.INPUT_RANGE_PM_1_V

                elif chan.range == Range.PM_5_V:
                    r = ats.INPUT_RANGE_PM_5_V

                if r < 0:
                    raise ValueError(f"Range is set to an unknown value: {chan.range}")

                impedance = (
                    ats.IMPEDANCE_50_OHM
                    if chan.termination == Termination.OHM_50
                    else ats.IMPEDANCE_1M_OHM
                )
                board.internal.inputControlEx(  # type: ignore
                    channel=ats.channels[i],
                    coupling=coupling,
                    inputRange=r,
                    impedance=impedance,
                )

        board.internal.setTriggerOperation(  # type: ignore
            ats.TRIG_ENGINE_OP_J,
            ats.TRIG_ENGINE_J,
            ats.TRIG_EXTERNAL,
            ats.TRIGGER_SLOPE_POSITIVE,
            self._trigger_level,
            ats.TRIG_ENGINE_K,
            ats.TRIG_DISABLE,
            ats.TRIGGER_SLOPE_POSITIVE,
            128,
        )

        board.internal.setExternalTrigger(  # type: ignore
            ats.DC_COUPLING,
            ats.ETR_5V,
        )

        trigger_delay_samples = 0
        board.internal.setTriggerDelay(trigger_delay_samples)  # type: ignore

        trigger_timeout_samples = self._trigger_timeout * self._sample_rate
        board.internal.setTriggerTimeOut(trigger_timeout_samples)  # type: ignore

    def _allocate_buffers(self, board: CombinedBoard, board_idx: int):
        channel_count = board.info.count_enabled()
        samples_per_buffer = self.samples_per_buffer

        channels = 0

        for i in range(len(board.info.channels)):
            if board.info.channels[i].enabled:
                channels += ats.channels[i]

        # Compute the number of bytes per record and per buffer
        _, bitsPerSample = board.internal.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        bytesPerBuffer = bytesPerSample * samples_per_buffer * channel_count

        self._sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            self._sample_type = ctypes.c_uint16

        for _ in range(self._num_buffers):
            self._buffers[board_idx].append(
                ats.DMABuffer(
                    board.internal.handle,
                    self._sample_type,
                    bytesPerBuffer,
                )
            )

        board.internal.beforeAsyncRead(  # type: ignore
            channels,
            0,
            self._samples_per_record,
            self._records_per_buffer,
            self._records_per_acquisition,
            self._adma_flags,
        )

        for buf in self._buffers[board_idx]:
            board.internal.postAsyncBuffer(buf.addr, buf.size_bytes)  # type: ignore

    def _acquire_data(self):
        start = time.time()

        try:
            self._boards[0].internal.startCapture()

            self.set_aux_out(True)

            i = 0

            while i < self._num_buffers and self.module_state() == "locked":
                self._data_transfer_loop(i)
                i += 1

        finally:
            for b in self._boards:
                b.internal.abortAsyncRead()

        self.set_aux_out(False)

        transfer_time = time.time() - start
        self.log.info(f"Data collection finished in: {transfer_time}")

    def _acquire_live_data(self):
        try:
            while self.module_state() == "locked":
                self._boards[0].internal.startCapture()

                self.set_aux_out(True)

                i = 0

                while i < self._num_buffers:
                    self._data_transfer_loop(i)
                    i += 1

        finally:
            for b in self._boards:
                b.internal.abortAsyncRead()

    def _data_transfer_loop(self, i: int):
        for b in range(len(self._boards)):
            timeout_ms = 5000
            buf = self._buffers[b][i]

            self._boards[b].internal.waitAsyncBufferComplete(buf.addr, timeout_ms)  # type: ignore

            # TODO: check if this needs a .copy() (or not)
            # Maybe do the copy on the other end
            self.sigNewData.emit(buf.buffer.copy())  # type: ignore

            self._boards[b].internal.postAsyncBuffer(  # type: ignore
                buf.addr,
                buf.size_bytes,  # type: ignore # TODO: This might be wrong / empty. CHECK
            )
