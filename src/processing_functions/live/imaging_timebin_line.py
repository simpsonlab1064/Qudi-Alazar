__all__ = ["imaging_timebin_line"]

import numpy as np
import numpy.typing as npt
from typing import Optional, cast  # type: ignore

from qudi.logic.piezo_logic import PiezoExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,
)


# Adaptive timing helpers
_prev_m: float | None = None
_prev_row_cache: dict[int, npt.NDArray[np.float64]] = {}




# def _imaging_timebin_line(
#     data: ProcessedData,
#     buf: npt.NDArray[np.int_],
#     settings: PiezoExperimentSettings,
#     buffer_index: int,
#     board_index: int,
#     boards: list[BoardInfo],
# ) -> ProcessedData:
#     # geometry
#     h = int(settings.height)
#     w = int(settings.width)

#     # channel layout
#     num_enabled = boards[board_index].count_enabled()
#     total_enabled = [b.count_enabled() for b in boards]
#     out_base = int(np.sum(total_enabled[:board_index]))

#     # persistent image allocation once per run
#     if buffer_index == 0 and board_index == 0:
#         data = ProcessedData(
#             data=[np.zeros((h, w), dtype=np.float64)
#                   for _ in range(int(np.sum(total_enabled)))]
#         )


#     # ADC centering and polarity

#     inv = bool(getattr(settings, "invert_polarity", False))

#     def _row_from_diff(seg: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.float64]:
#         """Resample to exactly 2*w samples (2 pulses per pixel) then average pairs."""
#         L = int(seg.size)
#         if L <= 1:
#             return np.zeros(w, dtype=np.float64)
#         target = 2 * w
#         xs = np.linspace(0.0, float(L - 1), num=target, endpoint=True, dtype=np.float64)
#         base = np.arange(L, dtype=np.float64)
#         y = np.interp(xs, base, seg.astype(np.float64, copy=False))
#         return y.reshape(w, 2).mean(axis=1)


#     ch_out = 0
#     for ch in boards[board_index].channels:
#         if not ch.enabled:
#             continue

#         # deinterleave this channel
#         line = np.asarray(buf[ch_out::num_enabled], dtype=np.float64)
#         # dynamic centering removes DC offset drift from preamp
#         line = -(line - float(np.mean(line))) if inv else (line - float(np.mean(line)))
#         n = int(line.size)
#         # quick numeric sanity check on first few buffers
#         if buffer_index < 5 and board_index == 0 and ch_out == 0:
#             print(f"[RAW] min={line.min():.0f} max={line.max():.0f} mean={line.mean():.1f}")

#         if n < 512:
#             return data

#         # ----- robust split between trace and retrace from slope flip -----
#         d = np.diff(line)
#         m0 = n // 2
#         win = max(64, n // 8)          # search ±12.5% around the center
#         lo = max(1, m0 - win)
#         hi = min(n - 2, m0 + win)

#         # strongest + → − slope change (triangle apex)
#         best_k, best_score = m0, -1.0
#         for k in range(lo, hi):
#             if d[k - 1] > 0 and d[k] <= 0:
#                 score = float(abs(d[k - 1]) + abs(d[k]))
#                 if score > best_score:
#                     best_score = score
#                     best_k = k

#         turn_idx = int(best_k)
#         m = 1024




#         # use trace only (left half) to keep 2 pulses/pixel total
#         trace = line[:m]

#         # separate two pulses per pixel with trimming + micro-alignment
#         length = int(m)
#         if (length & 1) == 1:
#             length -= 1
#         tr = trace[:length]
#         even = tr[0::2]
#         odd  = tr[1::2]

#         # small shift alignment between even/odd streams (compensates 0–2 sample skew)
#         best_s, best_c = 0, -1.0
#         for s in (-2, -1, 0, 1, 2):
#             if s >= 0:
#                 a = even[: even.size - s]
#                 b = odd[s:]
#             else:
#                 a = even[-s:]
#                 b = odd[: odd.size + s]
#             if a.size == 0 or b.size == 0:
#                 continue
#             c = float(np.dot(a - a.mean(), b - b.mean()))
#             if c > best_c:
#                 best_c, best_s = c, s

#         if best_s > 0:
#             odd = odd[best_s:]; even = even[: even.size - best_s]
#         elif best_s < 0:
#             even = even[-best_s:]; odd = odd[: odd.size + best_s]

#         N = int(min(even.size, odd.size))
#         even = even[:N]; odd = odd[:N]
#         diff_trace = even - odd


#         # light smoothing before binning
#         diff_trace = np.convolve(diff_trace, np.ones(3)/3, mode="same")


#         row = _row_from_diff(diff_trace, w)



#         # normalize per-row mean for consistent dynamic range
#         row -= np.mean(row)


#         # center and amplify contrast for visibility
#         row = row.astype(np.float64, copy=False)
#         row = (row - float(np.mean(row))) * 10.0
#         # if contrast looks inverted, temporarily flip sign:
#         # row = -(row)


#         # diagnostics (first few records)
#         if buffer_index < 10 and board_index == 0 and ch_out == 0:
#             spp_est = max(1, int(m // w))
#             print(f"[ADAPT] n={n} m_use={m} spp={spp_est} rows_written=1")

#         # line-to-line pixel alignment (±8 px) to remove diagonal streaks
#         out_idx = out_base + ch_out
#         prev: Optional[npt.NDArray[np.float64]] = _prev_row_cache.get(out_idx)

#         if prev is not None and prev.size == row.size:
#             xa = row - float(np.mean(row))
#             xb = prev - float(np.mean(prev))
#             max_shift = 64
#             best_s, best_c = 0, -1.0
#             for s in range(-max_shift, max_shift + 1):
#                 a0 = max(0, s); a1 = w + min(0, s)
#                 b0 = max(0, -s); b1 = w - max(0, s)
#                 c = float(np.dot(xa[a0:a1], xb[b0:b1]))
#                 if c > best_c:
#                     best_c, best_s = c, s
#             if best_s:
#                 row = np.roll(row, -best_s)
#         # remember this row for the next alignment step
#         _prev_row_cache[out_idx] = row.copy()

#         _prev_row_cache[out_idx] = (
#             0.8 * _prev_row_cache[out_idx] + 0.2 * row
#             if out_idx in _prev_row_cache
#             else row.copy()
#         )



#         # ----- write exactly one row per record -----
        
#         r = buffer_index % h
#         data.data[out_idx][r, :] = row


#         ch_out += 1

#     return data


def _imaging_timebin_line(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    h = settings.height
    w = settings.width
    num_enabled = boards[board_index].count_enabled()
    total_enabled = [b.count_enabled() for b in boards]

    # persistent image allocation once per run
    if buffer_index == 0 and board_index == 0:
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64)
                  for _ in range(int(np.sum(total_enabled)))]
        )

    buffer_index = buffer_index % 512

    rep = [buf[i] + buf[i+1] for i in range(0, len(buf)-1, 2)]
    rep = rep[::-1] if buffer_index % 2 == 0 else rep   
    data.data[board_index][buffer_index, :] = rep[:]

    return data


imaging_timebin_line = LiveProcessingInterface[PiezoExperimentSettings].from_function(
    _imaging_timebin_line
)
