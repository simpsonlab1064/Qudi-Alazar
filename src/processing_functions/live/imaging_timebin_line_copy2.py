__all__ = ["imaging_timebin_line"]

import numpy as np
import numpy.typing as npt
from typing import cast  # put at top of file with other imports #type:ignore
from qudi.logic.piezo_logic import PiezoExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,

)


def _imaging_timebin_line(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    # image shape
    h = int(settings.height)
    w = int(settings.width)

    # enabled channel counts
    num_enabled = boards[board_index].count_enabled()
    total_enabled = [b.count_enabled() for b in boards]
    out_base = int(np.sum(total_enabled[:board_index]))

    # first-call init for output images
    if buffer_index == 0 and board_index == 0:
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64)
                  for _ in range(int(np.sum(total_enabled)))]
        )

    # row index for this buffer
    row_idx = buffer_index % h #type:ignore

    

    # ADC centering and optional polarity inversion
    mid = float(getattr(settings, "adc_midcode", 32768.0))
    inv = bool(getattr(settings, "invert_polarity", False))

    ch_out = 0
    for ch in boards[board_index].channels:
        if not ch.enabled:
            continue

        # de-interleave this channel and center
        line = np.asarray(buf[ch_out::num_enabled], dtype=np.float64)
        line = line - mid
        if inv:
            line = -line

        n = line.size
        if n < 256:
            return data

               # split record into two half-lines
        turn_idx = n // 2
        m = int(min(turn_idx, n - turn_idx))
        if m < 64:
            return data

        trace = line[:m]
        retr  = line[turn_idx:turn_idx + m][::-1]  # retrace reversed to make left→right

        # helper: bin any 1D vector to width w using integer spp, truncate only
        def bin_to_width(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: #type:ignore

            L = int(seg.size)
            spp = max(1, L // w)      # integer samples per pixel
            need = spp * w
            seg2 = seg[:need]
            out = seg2.reshape(w, spp).mean(axis=1)
            return cast(npt.NDArray[np.float64], out)


        # average trace and retrace (two pulses per pixel → one row)
        avg_line = 0.5 * (trace + retr)

        # bin averaged waveform to image width
        spp = max(1, avg_line.size // w)
        need = spp * w
        seg = avg_line[:need]
        row_line = seg.reshape(w, spp).mean(axis=1)

        # write one averaged row per record
        out_idx = out_base + ch_out
        if row_idx < h:
            data.data[out_idx][row_idx, :] = row_line

        if buffer_index < 6 and board_index == 0 and ch_out == 0:
            print(f"[ROW] b={buffer_index} row={row_idx} n={n} turn_idx={turn_idx} spp={spp}")


        ch_out += 1

    return data



imaging_timebin_line = LiveProcessingInterface[PiezoExperimentSettings].from_function(
    _imaging_timebin_line
)
