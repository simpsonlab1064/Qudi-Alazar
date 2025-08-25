from __future__ import annotations

__all__ = ["PiezoStageInterface"]

from abc import abstractmethod

from qudi.core.module import Base  # type: ignore

from PySide2 import QtCore


byteorder: str = "little"
# TODO: Camryn: check if this is correct
# Ask Mark if it's the same as the galvo-galvo box.
# The other choice is "big"


class PiezoScanSettings:
    clk: float  # in Hz, pulled from hardware config
    enable_polarity: int  # 0 for LOW 1 for HI, pulled from config
    wave_f_mode: int  # 0 for triangular, 1 for ramp, pulled from config
    a_wave_on: int
    a_wave_off: int
    fast_wave_b_pulses: int
    fast_wave_ramp_steps: int
    fast_wave_scans_per_slow: int
    slow_wave_ramp_steps: int
    slow_wave_scans_per_trigger: int
    slow_wave_enable_mode: bool

    def __init__(
        self,
        a_wave_on: int = 1,
        a_wave_off: int = 1,
        fast_wave_b_pulses: int = 1,
        fast_wave_ramp_steps: int = 8,
        fast_wave_scans_per_slow: int = 1,
        slow_wave_ramp_steps: int = 8,
        slow_wave_scans_per_trigger: int = 1,
        slow_wave_enable_mode: bool = False,
    ):
        self.a_wave_on = a_wave_on
        self.a_wave_off = a_wave_off
        self.fast_wave_b_pulses = fast_wave_b_pulses
        self.fast_wave_ramp_steps = fast_wave_ramp_steps
        self.fast_wave_scans_per_slow = fast_wave_scans_per_slow
        self.slow_wave_ramp_steps = slow_wave_ramp_steps
        self.slow_wave_scans_per_trigger = slow_wave_scans_per_trigger
        self.slow_wave_enable_mode = slow_wave_enable_mode
        self.clk = 200
        self.enable_polarity = 1
        self.wave_f_mode = 0

    def __str__(self):
        return f"clk: {self.clk}, a_on: {self.a_wave_on}, a_off: {self.a_wave_off}, fast_b_pulses: {self.fast_wave_b_pulses}, ..."

    def __repr__(self):
        return f"<PiezoScanSettings clk: {self.clk}, a_on: {self.a_wave_on}, a_off: {self.a_wave_off}, fast_b_pulses: {self.fast_wave_b_pulses}, ...>"

    def to_bytes(
        self,
        fast_v_max: float,
        fast_v_min: float,
        slow_v_max: float,
        slow_v_min: float,
    ) -> bytes:
        def conversion(v: float) -> int:
            # TODO: Camryn: Check that this is correct,
            # Mark will know the answer for how to convert voltages to the
            # range he's using. This is assuming single-polarity with a max range
            # of 10 V.
            return round(v * 4095 / 10)

        return (
            self._to_bytes(3)  # Code for download + arm
            + self._to_bytes(self.enable_polarity)
            + self._to_bytes(self.wave_f_mode)
            + self._to_bytes(self.a_wave_on)
            + self._to_bytes(self.a_wave_off)
            + self._to_bytes(self.fast_wave_b_pulses)
            # TODO: Camryn: in document is labelled "WaveSreq". I don't know what that means
            # + self._to_bytes(0)    this was the previous version. camryn made new line below (line 83)
            + self._to_bytes(0 if self.slow_wave_enable_mode else self.slow_wave_scans_per_trigger)
            # TODO: Camryn: check -- "WaveFpixel" is how this is labelled in excel
            # but in the word doc it's ramp_steps (same for slow below)
            + self._to_bytes(self.fast_wave_ramp_steps)
            + self._to_bytes(conversion(fast_v_max))
            + self._to_bytes(conversion(fast_v_min))
            # TODO: Camryn: check -- "WaveSpixel"
            + self._to_bytes(self.slow_wave_ramp_steps)  #slow-axis pixel/step count
            + self._to_bytes(conversion(slow_v_max))
            + self._to_bytes(conversion(slow_v_min))
            + self._to_bytes(self.fast_wave_scans_per_slow)
            # CHECKSUM ADDED LATER
        )

    @staticmethod
    def _to_bytes(val: int) -> bytes:
        return (val).to_bytes(2, byteorder=byteorder)  # pyright: ignore[reportArgumentType]

    @staticmethod
    def representer_func(instance: PiezoScanSettings) -> list[object]:
        return [
            instance.a_wave_on,
            instance.a_wave_off,
            instance.fast_wave_b_pulses,
            instance.fast_wave_ramp_steps,
            instance.slow_wave_ramp_steps,
            instance.slow_wave_scans_per_trigger,
            instance.slow_wave_enable_mode,
        ]

    @staticmethod
    def constructor_func(yaml_data: list[object]) -> PiezoScanSettings:
        return PiezoScanSettings(
            a_wave_on=int(yaml_data[0]),  # type: ignore
            a_wave_off=int(yaml_data[1]),  # type: ignore
            fast_wave_b_pulses=int(yaml_data[2]),  # type: ignore
            fast_wave_ramp_steps=int(yaml_data[3]),  # type: ignore
            slow_wave_ramp_steps=int(yaml_data[4]),  # type: ignore
            slow_wave_scans_per_trigger=int(yaml_data[5]),  # type: ignore
            slow_wave_enable_mode=bool(yaml_data[6]),
        )


class PiezoStageInterface(Base):
    sigStageArmed = QtCore.Signal()

    _settings = PiezoScanSettings()

    @abstractmethod
    def on_activate(self) -> None:
        self.update_settings(self._settings)

    @abstractmethod
    def on_deactivate(self) -> None:
        self.end_scan()
        self.module_state.unlock()

    @property
    @abstractmethod
    def running(self) -> bool:
        pass

    @property
    @abstractmethod
    def get_settings(self) -> PiezoScanSettings:
        pass

    @QtCore.Slot(object)  # type: ignore
    @abstractmethod
    def update_settings(self, settings: PiezoScanSettings):
        pass

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def download_and_arm(self):
        pass

    @QtCore.Slot()  # type: ignore
    @abstractmethod
    def end_scan(self):
        pass
