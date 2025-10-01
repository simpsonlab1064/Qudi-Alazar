__all__ = ["AlazarImageGui"]

from qudi.core.module import GuiBase  # type: ignore
from qudi.core.connector import Connector  # type: ignore
from typing import cast

from qudi.logic.base_alazar_logic import (
    BaseExperimentSettings,
    DisplayData,
    DisplayType,
)

from PySide2 import QtCore, QtWidgets
import pyqtgraph as pg  # type: ignore



pg.setConfigOption("useOpenGL", True)  # type: ignore


class AlazarImageDisplayWindow(QtWidgets.QMainWindow):
    sigSelectChannel = QtCore.Signal(int)
    _data: list[DisplayData]
    _selected_idx: int = 0

    def __init__(self, settings: BaseExperimentSettings, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.setWindowTitle("Alazar Image Display")

        self._data = []
        self._selected_idx = 0

        # Main image widget
        self.image: pg.ImageView = pg.ImageView()

        # Right-side controls
        self.data_selection = QtWidgets.QComboBox()
        self.data_selection.currentIndexChanged.connect(self._select_data)  # type: ignore

        # Simple scale controls
        self._fixed_levels: tuple[float, float] | None = None
        self.freeze_btn = QtWidgets.QPushButton("Freeze scale")
        self.auto_btn = QtWidgets.QPushButton("Auto scale")
        self.freeze_btn.clicked.connect(self._freeze_levels)  # type: ignore
        self.auto_btn.clicked.connect(self._auto_levels)      # type: ignore

        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.image, 0, 0, 4, 4)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.data_selection)
        control_layout.addWidget(self.freeze_btn)
        control_layout.addWidget(self.auto_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout, 0, 5, 5, 1)
        layout.setColumnStretch(1, 1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initial state
        self.update_data(self._data)


    def update_data(self, data: list[DisplayData]):
        self._data = data

        if not self.data_selection.count() == len(data):
            self.data_selection.clear()
            self._selected_idx = 0

            for i, e in enumerate(self._data):
                if e.type == DisplayType.IMAGE:
                    self.data_selection.insertItem(i, e.label, e.label)

        else:
            # Assume the channels haven't changed meaning if they are of the
            # same length (good assumption?)
            if len(self._data) > 0:
                self._select_data(self._selected_idx)

        if self.data_selection.count() > self._selected_idx:
            self._select_data(self._selected_idx)

    def _select_data(self, idx: int):
        self._selected_idx = idx
        self._apply_levels()

    def _freeze_levels(self):
        """Lock the color scale to the current histogram range and pin the bar."""
        lo, hi = self.image.ui.histogram.getLevels()
        self._fixed_levels = (float(lo), float(hi))
        self._lock_hist_range(float(lo), float(hi))
        self._apply_levels()


    def _auto_levels(self):
        """Return to auto-leveling and release the bar."""
        self._fixed_levels = None
        self._unlock_hist_range()
        self._apply_levels()

    def _lock_hist_range(self, lo: float, hi: float) -> None:
        """Fix the histogram view so the color bar does not slide."""
        h = self.image.ui.histogram
        try:
            h.vb.disableAutoRange()              # stop viewbox autorange
        except Exception:
            pass
        try:
            h.setHistogramRange(lo, hi)          # pin bar to chosen range
        except Exception:
            # Fallback for older pyqtgraph: force the viewbox range directly
            h.vb.setYRange(lo, hi, padding=0)

    def _unlock_hist_range(self) -> None:
        """Return the histogram view to normal behavior."""
        h = self.image.ui.histogram
        try:
            h.vb.enableAutoRange()
        except Exception:
            pass

    def _apply_levels(self):
        """Render the current image using fixed levels when set, else auto."""
        if not self._data or self._selected_idx >= len(self._data):
            return

        arr = self._data[self._selected_idx].data  # type: ignore

        if self._fixed_levels is not None:
            lo, hi = self._fixed_levels
            self.image.setImage(arr, autoLevels=False)  # type: ignore[arg-type]
            cast(pg.ImageItem, self.image.getImageItem()).setLevels((lo, hi)) # type: ignore
            self.image.ui.histogram.setLevels(lo, hi)
            self._lock_hist_range(lo, hi)              # keep bar pinned
        else:
            self.image.setImage(arr)  # type: ignore[arg-type]



class AlazarImageGui(GuiBase):
    """
    This GUI displays anything that it thinks is "image-like". It expects data
    to be provided as a list[np.ndarray] and will give the option to plot any
    data that is
    """

    _logic = Connector(name="alazar_logic", interface="BaseAlazarLogic")

    _data: list[DisplayData] = []
    _settings: BaseExperimentSettings

    def on_activate(self):
        self._settings: BaseExperimentSettings = self._logic().experiment_info
        self._mw = AlazarImageDisplayWindow(self._settings)

        self._logic().sigImageDataUpdated.connect(self._update_data)

        self.show()

    def on_deactivate(self):
        self._mw.close()

    def show(self):
        self._mw.show()

    @QtCore.Slot(object)  # type: ignore
    def _update_data(self, data: list[DisplayData]):
        self._data.clear()
        for d in data:
            if d.type == DisplayType.IMAGE:
                self._data.append(d)

        self._mw.update_data(self._data)
