__all__ = ["PiezoStage"]

from PySide2 import QtCore
from qudi.core.configoption import ConfigOption  # type: ignore

import serial
import time

from typing import Literal

BYTEORDER: Literal["little"] = "little"   # silence type checker; same as your current value



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

    _slow_wave_enable_mode: bool = ConfigOption(name="slow_wave_enable_mode", default=False, missing="warn")  # type: ignore

    _fast_v_max: float = ConfigOption(name="fast_v_max", default=10, missing="warn")  # type: ignore

    _fast_v_min: float = ConfigOption(name="fast_v_min", default=2, missing="warn")  # type: ignore

    _slow_v_max: float = ConfigOption(name="slow_v_max", default=10, missing="warn")  # type: ignore

    _slow_v_min: float = ConfigOption(name="slow_v_min", default=2, missing="warn")  # type: ignore

    _com: str = ConfigOption(name="com", missing="error")  # type: ignore

    _baud: int = ConfigOption(name="baud", default=9600, missing="warn")  # type: ignore

    _protocol: str = ConfigOption(name="protocol", default="legacy", missing="info")  # type: ignore

    # run in separate thread
    _threaded = False

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        super().on_activate()
        self._serial = serial.Serial(
            self._com,
            self._baud,
            timeout=2,  # Just to prevent infinite hangs
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
        # For legacy protocol only; GGV1 derives timing from external 128 kHz, not this value
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
                # Build packet per GGSerialCOMM1.xlsx (Data Code 3 = ARM)
                # Field order (each is 2 bytes, LB–HB):
                # 1-2  Data Code = 3
                # 3-4  EnableMode: 0=Level, 1=Trig
                # 5-6  Enable Polarity: 0=LOW, 1=HIGH
                # 7-8  Fast Dac Mode: 0=TRI, 1=SINE
                # 9-10 Fast Dac Steps (pixels per line)
                # 11-12 Center Mirrors: 0=Normal, 1=Center
                # 13-14 Fast Dac Scans (fast scans per slow step)
                # 15-16 Slow Dac Mode: 0=Saw, 1=Ramp
                # 17-18 Slow Dac Steps (lines per frame)
                # 19-20 Slow Dac Scans (frames)
                # 21-22 CheckSum (sum of bytes 1–20)
                def u16(x: int) -> bytes:
                    return int(x).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]

                data_code = u16(3)
                enable_mode = u16(1 if getattr(self._settings, "slow_wave_enable_mode", False) else 0)
                enable_pol = u16(int(self._enable_polarity))  # 0=LOW, 1=HIGH

                # Map our wave_f_mode (0=ramp, 1=triangle) to controller (0=TRI, 1=SINE).
                fast_mode = u16(0 if int(self._wave_f_mode) == 1 else 1) 

                fast_steps = u16(int(getattr(self._settings, "fast_wave_ramp_steps", 512)))
                center_mirrors = u16(0)  # 0=Normal
                fast_scans = u16(int(getattr(self._settings, "fast_wave_scans_per_slow", 1)))

                # Slow axis: use ramp stepping
                slow_mode = u16(1)
                slow_steps = u16(int(getattr(self._settings, "slow_wave_ramp_steps", 512)))
                slow_scans = u16(int(getattr(self._settings, "slow_wave_scans_per_trigger", 1)))

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
                packet = packet_wo_chk + chk.to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]
            else:
                # Legacy behavior (unchanged): delegate to settings.to_bytes(...)
                packet = self._settings.to_bytes(
                    fast_v_max=self._fast_v_max,
                    fast_v_min=self._fast_v_min,
                    slow_v_max=self._slow_v_max,
                    slow_v_min=self._slow_v_min,
                )
            # ... after building packet_wo_chk ...
                chk = sum(packet)  # int
                packet = packet + ((chk & 0xFFFF).to_bytes(2, byteorder=BYTEORDER, signed=False))


                fast_steps_i = int(getattr(self._settings, "fast_wave_ramp_steps", 512))
                fast_scans_i = int(getattr(self._settings, "fast_wave_scans_per_slow", 1))
                slow_steps_i = int(getattr(self._settings, "slow_wave_ramp_steps", 512))
                slow_scans_i = int(getattr(self._settings, "slow_wave_scans_per_trigger", 1))
                enable_mode_i = 1 if getattr(self._settings, "slow_wave_enable_mode", False) else 0
                enable_pol_i = int(self._enable_polarity)
                fast_mode_i  = 0 if int(self._wave_f_mode) == 1 else 1  # 0=TRI, 1=SINE
                chk16 = (chk & 0xFFFF)

                self.log.info(
                    f"[GGV1 ARM] enable_mode={enable_mode_i} enable_polarity={enable_pol_i} "
                    f"fast_mode={fast_mode_i} fast_steps={fast_steps_i} fast_scans={fast_scans_i} "
                    f"slow_mode=1 slow_steps={slow_steps_i} slow_scans={slow_scans_i} checksum=0x{chk16:04X}"
                )


            self.log.info("PiezoStage[GGV1]: ignoring YAML 'clock'; timing derives from external ECLK via EXT CLOCK INPUT.")

            self._send_command(packet)
            self._receive_data()

            self.sigStageArmed.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    def end_scan(self):
        if self.module_state() == "locked":
            if getattr(self, "_protocol", "legacy") == "ggv1":
                # UN-ARM per GGSerialCOMM1.xlsx:
                # 1-2 Data Code = 6
                # 3-4 CheckSum  = sum(bytes 1..2)  (16-bit little-endian)
                data = (6).to_bytes(2, byteorder=BYTEORDER, signed=False)
                chk  = (sum(data) & 0xFFFF).to_bytes(2, byteorder=BYTEORDER, signed=False)
                packet = data + chk

            else:
                # legacy path unchanged
                packet  = (6).to_bytes(2, byteorder=BYTEORDER)
                packet += (6).to_bytes(2, byteorder=BYTEORDER)



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

        resp_chksum = int.from_bytes(resp[4:6], byteorder=BYTEORDER)
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
        chk = int.from_bytes(bys[6:], byteorder=BYTEORDER)
        return chk == self._calc_checksum(bys)
    
    
