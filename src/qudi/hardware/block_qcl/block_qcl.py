# -*- coding: utf-8 -*-

__all__ = ["PiezoDummy"]

from PySide2 import QtCore

from qudi.interface.piezo_stage_interface import (
    PiezoStageInterface,
    PiezoScanSettings,
)


class PiezoDummy(PiezoStageInterface):
    """
    Dummy for controlling a piezo stage

    Example config for copy-paste:

    piezo_stage_dummy:
        module.Class: 'piezo_stage.piezo_dummy.PiezoStageDummy'
    """

    # run in separate thread
    _threaded = True

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore

    def on_activate(self) -> None:
        super().on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()

    @property
    def running(self) -> bool:
        return self.module_state() == "locked"

    @property
    def get_settings(self) -> PiezoScanSettings:
        return self._settings

    @QtCore.Slot(object)  # type: ignore
    def update_settings(self, settings: PiezoScanSettings):
        self._settings = settings

    @QtCore.Slot()  # type: ignore
    def download_and_arm(self):
        if self.module_state() != "locked":
            self.module_state.lock()
            self.sigStageArmed.emit()  # type: ignore

    @QtCore.Slot()  # type: ignore
    def end_scan(self):
        if self.module_state() == "locked":
            self.module_state.unlock()
