from __future__ import annotations

__all__ = ["photothermal_avi"]

import numpy as np
import numpy.typing as npt

from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,
)
from qudi.logic.photothermal_avi_logic import PhotothermalAVISettings


def _photothermal_avi(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PhotothermalAVISettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    h = int(settings.height)
    w = int(settings.width)

    num_enabled = boards[board_index].count_enabled()
    total_enabled = [b.count_enabled() for b in boards]
    out_base = int(np.sum(total_enabled[:board_index]))

    # Allocate output arrays once at the start of an acquisition
    if buffer_index == 0 and board_index == 0:
        data = ProcessedData(
            data=[
                np.zeros((h, w), dtype=np.float64)
                for _ in range(int(np.sum(total_enabled)))
            ]
        )

    row_index = buffer_index % h
    ch_out = 0


    # Only keep the trace half of each fast cycle (one row per triangle)
    if row_index >= h:
        return data


    for ch in boards[board_index].channels:
        if not ch.enabled:
            continue

        # Full triangle: 1024 samples
        line = np.asarray(buf[ch_out::num_enabled], dtype=np.float64)

        half = line.size // 2

        # Use ONLY the trace half; discard retrace to remove duplicate image
        trace = line[:half]

        # Use 512 output pixels
        row = trace[:w]




        out_idx = out_base + ch_out
        data.data[out_idx][row_index, :] = row

        ch_out += 1

    return data


photothermal_avi = LiveProcessingInterface[PhotothermalAVISettings].from_function(
    _photothermal_avi
)
