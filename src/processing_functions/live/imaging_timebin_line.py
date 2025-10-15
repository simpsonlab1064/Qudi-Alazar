__all__ = ["imaging_timebin_line"]

import numpy as np
import numpy.typing as npt

from processing_functions.util.numpy_groupies.aggregate_numpy import aggregate  # type: ignore
from qudi.logic.piezo_logic import PiezoExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.raster_time_to_pix_num import raster_time_to_pix_num  # type: ignore
#from processing_functions.util.voltage_average_image import voltage_average_image
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,
)


def time_to_cols(
    num_samples: int,
    sample_rate_hz: float,
    width_px: int,
    fast_period_us: float,
    fast_phase: float,
) -> npt.NDArray[np.int_]:
    """
    Linear trace-only mapper: map the first width_px samples to columns 0..width_px-1.
    Ignores timing parameters intentionally for this diagnostic step.
    """
    n = int(min(num_samples, width_px))
    return np.arange(n, dtype=np.int_)


# def _imaging_timebin_line(
#     data: ProcessedData,
#     buf: npt.NDArray[np.int_],
#     settings: ImagingExperimentSettings,
#     buffer_index: int,
#     board_index: int,
#     boards: list[BoardInfo],
# ) -> ProcessedData:
#     """
#     One buffer = one full fast-axis line period: TRACE (L→R) + RETRACE (R→L).
#     We map the first half to columns 0..w-1 and the second half to columns w-1..0,
#     and write BOTH into the SAME image row.
#     """
#     h = int(settings.height)
#     w = int(settings.width)
#     #transpose = bool(getattr(settings, "transpose_image", False))

#     # only one Alazar channel enabled in this test
#     enabled = [c.enabled for c in boards[board_index].channels]
#     nchan = int(sum(1 for e in enabled if e))
#     if nchan <= 0:
#         raise ValueError("No enabled channels")

#     if nchan != 1:
#         ch = np.asarray(buf[0::nchan][: (buf.size // nchan)], dtype=np.float64)
#     else:
#         ch = np.asarray(buf, dtype=np.float64)


#         # Center around ADC midcode; keep your PMT negative sign (no inversion)
#     invert = bool(getattr(settings, "invert_polarity", False))
#     ch = (-ch) if invert else ch

#     # STEP 2: trim flyback and settle regions on each line
#     head = int(getattr(settings, "line_head_trim", 0))
#     tail = int(getattr(settings, "line_tail_trim", 0))
#     if head or tail:
#         if ch.size > (head + tail):
#             ch = ch[head: ch.size - tail]


#         # --- force one line period to be exactly 2*width samples ---
#         # --- force one line period to be exactly 2*width samples ---
#     expected = 2 * w
#     if ch.size != expected:
#         if ch.size > expected:
#             # symmetric center crop to preserve trace/retrace alignment
#             start = (ch.size - expected) // 2
#             ch = ch[start:start + expected]
#         else:
#             # symmetric edge padding to preserve alignment
#             deficit = expected - ch.size
#             left = deficit // 2
#             right = deficit - left
#             ch = np.pad(ch, (left, right), mode="edge")
#         if buffer_index < 6 and board_index == 0:
#             print(f"[FIX] ch len -> {ch.size} (expected {expected}), symmetric crop/pad applied")


#     # polarity: make brighter = positive (PMT-style)
# # polarity / offset: Alazar delivers signed int16 -> already centered at 0
#     mid = float(getattr(settings, "adc_midcode", 0.0))
#     invert = bool(getattr(settings, "invert_polarity", False))
#     ch = (-(ch - mid)) if invert else (ch - mid)


#     # First-time init of the image
#     if buffer_index == 0 and board_index == 0:
#         # NOTE: image is (h, w) internally; UI can transpose if desired
#         data = ProcessedData(data=[np.zeros((h, w), dtype=np.float64)])

#     ch_use = ch  # no trimming for now; must be exactly 2*width

#     if ch_use.size < 2:
#         return data

#     # Use TRACE only to avoid left/right duplication
#     half = int(ch_use.size // 2)
#     trace = ch_use[:half]

#     def _bin_to_w(x: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.float64]:
#         if x.size == 0:
#             return np.zeros(w, dtype=np.float64)
#         if x.size == 1:
#             return np.full(w, float(x[0]), dtype=np.float64)
#         src = np.arange(x.size, dtype=np.float64)
#         tgt = np.linspace(0.0, float(x.size - 1), w, dtype=np.float64)
#         return np.interp(tgt, src, x).astype(np.float64, copy=False)

#     row_line = _bin_to_w(trace, w)

#     # Optional: drop the first few rows to avoid top-of-frame corruption
#     rows_to_skip = int(getattr(settings, "frame_head_rows_skip", 0))
#     row_idx = int(buffer_index % h)
#     if row_idx < rows_to_skip:
#         # Fill with the median of the line to keep display stable
#         fill = float(np.median(row_line)) if row_line.size else 0.0
#         row_line = np.full(w, fill, dtype=np.float64)

#     img = data.data[0]
#     if not getattr(settings, "transpose_image", False):
#         img[row_idx, :] = row_line
#     else:
#         img[:, row_idx] = row_line

#     if buffer_index < 8 and board_index == 0:
#         lo, hi = np.percentile(row_line, [0.5, 99.5])
#         print(f"[ROW] buf={buffer_index} row={row_idx} trace_len={trace.size} ch_len={ch_use.size} p0.5={lo:.1f} p99.5={hi:.1f}")


#     return data


def _imaging_timebin_line(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    # External clock = 64 kHz
    # fast_wave_ramp_steps = 1024  → expected samples per line = 2*1024 = 2048 (trace + retrace)
    # Display width = 512 → bin trace to 512 columns → 2 pulses per pixel

    h = int(settings.height)
    w = int(settings.width)

    # number of enabled channels on the current board
    num_enabled = boards[board_index].count_enabled()
    total_enabled = [b.count_enabled() for b in boards]

    # initialize once on the very first buffer only
    if buffer_index == 0 and board_index == 0:
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64)
                for _ in range(int(np.sum(total_enabled)))]
        )


    row_idx = buffer_index % h
    i = 0

    for c in boards[board_index].channels:
        if c.enabled:
            # one channel’s interleaved line
            line = np.asarray(buf[i::num_enabled], dtype=np.float64)

            # Use TRACE only: first 2*w samples. For 2 pulses/pixel, 2*w = 1024 when w=512.
            need = 2 * w
            if line.size < need:
                return data
            trace = line[:need]

            # Average adjacent pairs -> w pixels
            row_line = trace.reshape(w, 2).mean(axis=1)

            # Serpentine writeback: reverse every other row
            if (buffer_index % 2) == 1:
                row_line = row_line[::-1]

            idx = int(np.sum(total_enabled[:board_index])) + i
            data.data[idx][row_idx, :] = row_line

            i += 1

    return data




imaging_timebin_line = LiveProcessingInterface[PiezoExperimentSettings].from_function(
    _imaging_timebin_line
)
