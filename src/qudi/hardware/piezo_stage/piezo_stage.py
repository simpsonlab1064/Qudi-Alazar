# -*- coding: utf-8 -*-

__all__ = ["PiezoStage"]

from PySide2 import QtCore
from qudi.core.configoption import ConfigOption  # type: ignore

import serial
import time


from qudi.interface.piezo_stage_interface import (
    PiezoStageInterface,
    PiezoScanSettings,
    byteorder,
)


class PiezoStage(PiezoStageInterface):
    """
    Hardware def. for controlling a piezo stage via an Arduino box fabricated by the
    Amy Facility.

    Example config for copy-paste:

    piezo_stage:
        module.Class: 'piezo_stage.piezo_stage.PiezoStage'
        options:
            clock: 208.3 # in Hz, clock frequency into the piezo control box
            enable_polarity: 1 # 0 for LO, 1 for HI
            wave_f_mode: 1 # 0 for ramp, 1 for triangle
            fast_v_max: 10 # in V
            fast_v_min: 2 # in V
            slow_v_max: 10 # in V
            slow_v_min: 2 # in V
            com: 'COM1' # string of com port we are attached to (or like '/dev/ttyUSB')
            baud: 9600 # baud rate of serial connection
    """

    # Declare static parameters that can/must be declared in the qudi configuration
    _clock: float = ConfigOption(name="clock", default=200, missing="warn")  # type: ignore

    _enable_polarity: int = ConfigOption(
        name="enable_polarity", default=1, missing="warn"
    )  # type: ignore

    _wave_f_mode: int = ConfigOption(name="wave_f_mode", default=1, missing="warn")  # type: ignore

    _fast_v_max: float = ConfigOption(name="fast_v_max", default=10, missing="warn")  # type: ignore

    _fast_v_min: float = ConfigOption(name="fast_v_min", default=2, missing="warn")  # type: ignore

    _slow_v_max: float = ConfigOption(name="slow_v_max", default=10, missing="warn")  # type: ignore

    _slow_v_min: float = ConfigOption(name="slow_v_min", default=2, missing="warn")  # type: ignore

    _com: str = ConfigOption(name="com", missing="error")  # type: ignore

    _baud: int = ConfigOption(name="baud", default=9600, missing="warn")  # type: ignore

    # run in separate thread
    _threaded = True

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        super().on_activate()
        self._serial = serial.Serial(
            self._com,
            self._baud,
            timeout=10, # Just to prevent infinite hangs
        )

        # Need to pause before connecting as opening serial port
        # resets the arduino
        time.sleep(2)

        resp = self._connect()

        if not resp:
            raise ValueError("Connection to Arduino control box failed!")

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._serial.close()

    @property
    def running(self) -> bool:
        return self.module_state() == "locked"

    @property
    def get_settings(self) -> PiezoScanSettings:
        return self._settings

    @QtCore.Slot(object)  # type: ignore
    def update_settings(self, settings: PiezoScanSettings):
        self._settings = settings
        self._settings.clk = self._clock
        self._settings.enable_polarity = self._enable_polarity
        self._settings.wave_f_mode = self._wave_f_mode

    @QtCore.Slot()  # type: ignore
    def download_and_arm(self):
        if self.module_state() != "locked":
            self.module_state.lock()
            packet = self._settings.to_bytes(
                fast_v_max=self._fast_v_max,
                fast_v_min=self._fast_v_min,
                slow_v_max=self._slow_v_max,
                slow_v_min=self._slow_v_min,
            )

            chk = sum(packet)

            packet = packet + (chk).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]

            self._send_command(packet)
            self._receive_data()

            self.sigStageArmed.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    def end_scan(self):
        if self.module_state() == "locked":
            packet = (6).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]
            packet += (6).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]
            self._send_command(packet)
            self.module_state.unlock()

    def _connect(self) -> bool:
        cmd = (1).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]
        chk = (1).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]
        packet = cmd + chk
        self._send_command(packet)

        return self._receive_data()

    def _send_command(self, packet: bytes):
        self._serial.write(packet)

    def _receive_data(self) -> bool:
        resp = self._serial.read(8)  # Always should return 8 bytes

        valid = self._valid_checksum(resp)

        if not valid:
            self.log.warning(f"Response had an invalid checksum. Response was: {resp}")
            return False

        resp_chksum = int.from_bytes(resp[4:6], byteorder=byteorder)  # pyright: ignore[reportArgumentType]
        if resp_chksum != 0:
            self.log.warning(
                f"Response indicates that an error ocurrred, code was: {resp_chksum}"
            )
            return False

        return True

    def _calc_checksum(self, bys: bytes) -> int:
        return sum(bys[0:-2])

    def _valid_checksum(self, bys: bytes) -> bool:
        if len(bys) != 8:
            self.log.warning(
                f"Response from Arduino had incorrect number of bytes! Length was: {len(bys)}"
            )
            return False
        chk = int.from_bytes(bys[6:], byteorder=byteorder)  # pyright: ignore[reportArgumentType]
        return chk == self._calc_checksum(bys)
