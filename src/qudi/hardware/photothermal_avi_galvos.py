from __future__ import annotations

__all__ = ["GalvoStage"]

from PySide2 import QtCore
from qudi.core.configoption import ConfigOption  # type: ignore
from qudi.interface.piezo_stage_interface import (
    PiezoStageInterface,
    PiezoScanSettings,
)


import serial
import time
from typing import Literal

BYTEORDER: Literal["little"] = "little"
byteorder: Literal["little"] = "little"









class GalvoStage(PiezoStageInterface):
    _clock: float = ConfigOption(name="clock", default=200, missing="warn")  # type: ignore
    _enable_polarity: int = ConfigOption(name="enable_polarity", default=1, missing="warn")  # type: ignore
    _wave_f_mode: int = ConfigOption(name="wave_f_mode", default=1, missing="warn")  # type: ignore
    _slow_wave_enable_mode: bool = ConfigOption(name="slow_wave_enable_mode", default=False, missing="warn")  # type: ignore
    _fast_v_max: float = ConfigOption(name="fast_v_max", default=10, missing="warn")  # type: ignore
    _fast_v_min: float = ConfigOption(name="fast_v_min", default=2, missing="warn")  # type: ignore
    _slow_v_max: float = ConfigOption(name="slow_v_max", default=10, missing="warn")  # type: ignore
    _slow_v_min: float = ConfigOption(name="slow_v_min", default=2, missing="warn")  # type: ignore
    _com: str = ConfigOption(name="com", missing="error")  # type: ignore
    _baud: int = ConfigOption(name="baud", default=9600, missing="warn")  # type: ignore
    _protocol: str = ConfigOption(name="protocol", default="legacy", missing="info")  # type: ignore

    _threaded = False

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        super().on_activate()
        self._serial = serial.Serial(
            self._com,
            self._baud,
            timeout=2,
        )
        time.sleep(2)
        resp = self._connect()
        if not resp:
            raise ValueError("Connection to Galvo control box failed.")

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
        if getattr(self, "_protocol", "legacy") != "ggv1":
            self._settings.clk = self._clock
        self._settings.enable_polarity = self._enable_polarity
        self._settings.wave_f_mode = self._wave_f_mode
        try:
            self._settings.slow_wave_enable_mode = bool(self._slow_wave_enable_mode)  # type: ignore[attr-defined]
        except AttributeError:
            self.log.warning("PiezoScanSettings lacks 'slow_wave_enable_mode'; proceeding without it.")

    @QtCore.Slot()  # type: ignore
    def download_and_arm(self):
        if self.module_state() != "locked":
            self.module_state.lock()

            if self._protocol == "ggv1":
                def u16(x: int) -> bytes:
                    return int(x).to_bytes(2, byteorder=byteorder)

                data_code = u16(3)
                enable_mode = u16(1 if getattr(self._settings, "slow_wave_enable_mode", False) else 0)
                enable_pol = u16(int(self._enable_polarity))
                fast_mode = u16(0 if int(self._wave_f_mode) == 1 else 1)
                fast_steps = u16(int(getattr(self._settings, "fast_wave_ramp_steps", 512)))
                center_mirrors = u16(0)
                fast_scans = u16(int(getattr(self._settings, "fast_wave_scans_per_slow", 1)))
                slow_mode = u16(1)
                slow_steps = u16(int(getattr(self._settings, "slow_wave_ramp_steps", 512)))
                slow_scans = u16(1)  # exactly one slow ramp per trigger


                packet_wo_chk = (
                    data_code
                    + enable_mode
                    + enable_pol
                    + fast_mode
                    + fast_steps
                    + center_mirrors
                    + fast_scans
                    + slow_mode
                    + slow_steps
                    + slow_scans
                )
                chk = sum(packet_wo_chk)
                packet = packet_wo_chk + chk.to_bytes(2, byteorder=byteorder)
            else:
                packet = self._settings.to_bytes(
                    fast_v_max=self._fast_v_max,
                    fast_v_min=self._fast_v_min,
                    slow_v_max=self._slow_v_max,
                    slow_v_min=self._slow_v_min,
                )
                chk = sum(packet)
                packet = packet + ((chk & 0xFFFF).to_bytes(2, byteorder=BYTEORDER, signed=False))

            self._send_command(packet)
            self._receive_data()
            self.sigStageArmed.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    def end_scan(self):
        if self.module_state() == "locked":
            if getattr(self, "_protocol", "legacy") == "ggv1":
                data = (6).to_bytes(2, byteorder=BYTEORDER, signed=False)
                chk = (sum(data) & 0xFFFF).to_bytes(2, byteorder=BYTEORDER, signed=False)
                packet = data + chk
            else:
                packet = (6).to_bytes(2, byteorder=BYTEORDER) + (6).to_bytes(2, byteorder=BYTEORDER)
            self._send_command(packet)
            self.module_state.unlock()

    def _connect(self) -> bool:
        cmd = (1).to_bytes(2, byteorder=byteorder)
        chk = (1).to_bytes(2, byteorder=byteorder)
        packet = cmd + chk
        self._send_command(packet)
        return self._receive_data()

    def _send_command(self, packet: bytes):
        self._serial.write(packet)

    def _receive_data(self) -> bool:
        resp = self._serial.read(8)
        if not self._valid_checksum(resp):
            self.log.warning(f"Response had an invalid checksum. Response was: {resp}")
            return False
        resp_chksum = int.from_bytes(resp[4:6], byteorder=BYTEORDER)
        if resp_chksum != 0:
            self.log.warning(
                f"Response indicates that an error occurred, code was: {resp_chksum}"
            )
            return False
        return True

    def _calc_checksum(self, bys: bytes) -> int:
        return sum(bys[0:-2])

    def _valid_checksum(self, bys: bytes) -> bool:
        if len(bys) != 8:
            self.log.warning(
                f"Response from Galvo box had incorrect number of bytes. Length was: {len(bys)}"
            )
            return False
        chk = int.from_bytes(bys[6:], byteorder=BYTEORDER)
        return chk == self._calc_checksum(bys)
