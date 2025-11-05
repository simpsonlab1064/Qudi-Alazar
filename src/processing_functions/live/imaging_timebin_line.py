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




def _estimate_half_period(line: npt.NDArray[np.float64]) -> int:
    n = int(line.size)
    if n < 128:
        return n // 2
    x = line - float(np.mean(line))
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    k0 = max(n // 4, 32)
    k1 = min(3 * n // 4, n - 32)
    idx_rel = int(np.argmax(ac[k0:k1]))
    return int(k0 + idx_rel)

def _smooth_m(m_new: int) -> int:
    global _prev_m
    if _prev_m is None:
        _prev_m = float(m_new)
    else:
        _prev_m = 0.9 * _prev_m + 0.1 * float(m_new)
    return int(round(_prev_m))



def _imaging_timebin_line(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    # geometry
    h = int(settings.height)
    w = int(settings.width)

    # channel layout
    num_enabled = boards[board_index].count_enabled()
    total_enabled = [b.count_enabled() for b in boards]
    out_base = int(np.sum(total_enabled[:board_index]))

    # persistent image allocation once per run
    if buffer_index == 0 and board_index == 0:
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64)
                  for _ in range(int(np.sum(total_enabled)))]
        )

    # row index managed outside to keep live display persistent
    row_idx = buffer_index % h

    # ADC centering and polarity
    mid = float(getattr(settings, "adc_midcode", 32768.0))
    inv = bool(getattr(settings, "invert_polarity", False))

    def bin_to_width(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        L = int(seg.size)
        if L <= 0:
            return np.zeros(w, dtype=np.float64)
        spp = max(1, L // w)              # integer samples per pixel
        need = spp * w
        seg2 = seg[:need]
        out = seg2.reshape(w, spp).mean(axis=1)
        return out.astype(np.float64, copy=False)

    ch_out = 0
    for ch in boards[board_index].channels:
        if not ch.enabled:
            continue

        # deinterleave this channel
        line = np.asarray(buf[ch_out::num_enabled], dtype=np.float64)
        line = -(line - mid) if inv else (line - mid)
        n = int(line.size)
        # quick numeric sanity check on first few buffers
        if buffer_index < 5 and board_index == 0 and ch_out == 0:
            print(f"[RAW] min={line.min():.0f} max={line.max():.0f} mean={line.mean():.1f}")

        if n < 512:
            return data

       # ----- adaptive timing and single-row reconstruction -----
        m_raw = _estimate_half_period(line)
        m = _smooth_m(m_raw)
        m = int(max(64, min(m, n // 2)))  # guardrails

        # use trace only to keep 2 pulses/pixel total (no retrace duplication)
        trace = line[:m]

        # separate two pulses per pixel with robust trimming + small shift alignment
        # ensure even length, then split into even/odd streams
        length = int(m)
        if (length & 1) == 1:
            length -= 1
        tr = trace[:length]
        even = tr[0::2]
        odd  = tr[1::2]


        # micro-alignment of odd vs even (compensate 0–2 sample skew)
        best_s = 0
        best_c = -1.0
        for s in (-2, -1, 0, 1, 2):
            if s >= 0:
                a = even[: even.size - s]
                b = odd[s:]
            else:
                a = even[-s:]
                b = odd[: odd.size + s]
            if a.size == 0 or b.size == 0:
                continue
            c = float(np.dot(a - a.mean(), b - b.mean()))
            if c > best_c:
                best_c = c
                best_s = s

        # apply best shift and equalize lengths
        if best_s > 0:
            odd = odd[best_s:]
            even = even[: even.size - best_s]
        elif best_s < 0:
            even = even[-best_s:]
            odd = odd[: odd.size + best_s]

        N = int(min(even.size, odd.size))
        even = even[:N]
        odd  = odd[:N]

        diff_trace = even - odd



        def bin_to_width(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            L = int(seg.size)
            if L <= 0:
                return np.zeros(w, dtype=np.float64)
            if L >= w:
                # floor spp so that spp * w <= L   (prevents reshape errors)
                spp = max(1, L // w)
                need = spp * w
                seg2 = seg[:need]
                out = seg2.reshape(w, spp).mean(axis=1)
                return out.astype(np.float64, copy=False)
            else:
                # rare case L < w: interpolate up to width
                xs = np.linspace(0.0, float(L - 1), num=w, endpoint=True, dtype=np.float64)
                base = np.arange(L, dtype=np.float64)
                out = np.interp(xs, base, seg.astype(np.float64, copy=False))
                return out.astype(np.float64, copy=False)

        # light smoothing before binning
        diff_trace = np.convolve(diff_trace, np.ones(3)/3, mode="same")


        row = bin_to_width(diff_trace)


        # normalize per-row mean for consistent dynamic range
        row -= np.mean(row)


        # center and amplify contrast for visibility
        row = row.astype(np.float64, copy=False)
        row = (row - float(np.mean(row))) * 50.0
        # if contrast looks inverted, temporarily flip sign:
        # row = -(row)


        # diagnostics (first few records)
        if buffer_index < 10 and board_index == 0 and ch_out == 0:
            spp_est = max(1, int(m // w))
            print(f"[ADAPT] n={n} m_raw={m_raw} m_use={m} spp={spp_est} rows_written=1")

        # line-to-line pixel alignment (±8 px) to remove diagonal streaks
        out_idx = out_base + ch_out
        prev: Optional[npt.NDArray[np.float64]] = _prev_row_cache.get(out_idx)

        if prev is not None and prev.size == row.size:
            xa = row - float(np.mean(row))
            xb = prev - float(np.mean(prev))
            max_shift = 8
            best_s, best_c = 0, -1.0
            for s in range(-max_shift, max_shift + 1):
                a0 = max(0, s); a1 = w + min(0, s)
                b0 = max(0, -s); b1 = w - max(0, s)
                c = float(np.dot(xa[a0:a1], xb[b0:b1]))
                if c > best_c:
                    best_c, best_s = c, s
            if best_s:
                row = np.roll(row, -best_s)
        # remember this row for the next alignment step
        _prev_row_cache[out_idx] = row.copy()


        # ----- write exactly one row per record -----
        
        r = buffer_index % h
        data.data[out_idx][r, :] = row


        ch_out += 1

    return data




imaging_timebin_line = LiveProcessingInterface[PiezoExperimentSettings].from_function(
    _imaging_timebin_line
)
