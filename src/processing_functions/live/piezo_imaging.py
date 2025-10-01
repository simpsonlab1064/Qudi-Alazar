from __future__ import annotations
from typing import cast 


__all__ = ["piezo_imaging"]

import numpy as np
import numpy.typing as npt

from qudi.logic.piezo_logic import PiezoExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,
)


# -------------------- helpers --------------------

def _slice_channel_block(
    buf: npt.NDArray[np.int_], nchan: int, chan_index: int
) -> npt.NDArray[np.float64]:
    total = int(buf.size)
    per_chan = total // nchan
    return np.asarray(buf[chan_index::nchan][:per_chan], dtype=np.float64)



def _map_two_rows(
    img: npt.NDArray[np.float64],
    row0: int,
    row1: int,
    trace: npt.NDArray[np.float64],
    retrace: npt.NDArray[np.float64],
    phase_frac: float,
    detrend: bool = True,
    normalize: bool = True,
) -> None:
    # optional phase roll on each line
    if phase_frac:
        w_loc = int(trace.size)
        shift_cols = int(round((phase_frac % 1.0) * w_loc))
        if shift_cols:
            trace = np.roll(trace, shift_cols)
            retrace = np.roll(retrace, shift_cols)

    # light denoise
    if trace.size >= 9:
        k = np.ones(9, dtype=np.float64) / 9.0
        trace = np.convolve(trace, k, mode="same")
        retrace = np.convolve(retrace, k, mode="same")

    def _prep(line: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = line.astype(np.float64, copy=False)
        if detrend:
            n = out.size
            n_edge = min(32, max(1, n // 16))
            left = float(np.median(out[:n_edge]))
            right = float(np.median(out[-n_edge:]))
            m = (right - left) / max(1.0, n - 1.0)
            b = left
            x = np.arange(n, dtype=np.float64)
            out = out - (m * x + b)
        if normalize:
            p1, p99 = np.percentile(out, [1.0, 99.0])
            scale = p99 - p1 if p99 > p1 else 1.0
            out = (out - p1) / scale - 0.5
        return out

    img[row0, : trace.size] = _prep(trace)
    if row1 != row0:
        img[row1, : retrace.size] = _prep(retrace)



def _estimate_column_shift( #type:ignore
    trace: npt.NDArray[np.float64],
    retrace: npt.NDArray[np.float64],
    max_shift: int = 64,
) -> int:
    """
    Estimate integer column shift between trace and retrace.
    Retrace provided in left->right order (already reversed if using serpentine).
    Returns shift in columns to apply with np.roll on BOTH lines.
    """
    w = int(min(trace.size, retrace.size))
    t = trace[:w].astype(np.float64, copy=False)
    r = retrace[:w].astype(np.float64, copy=False)

    # remove DC for robust correlation
    t = t - np.median(t)
    r = r - np.median(r)

    # limit search window
    m = int(min(max_shift, w // 4))
    if m <= 0:
        return 0

    # compute normalized cross-correlation for integer shifts
    best_shift = 0
    best_val = -1.0
    for s in range(-m, m + 1):
        if s < 0:
            a = t[-s:w]
            b = r[:w + s]
        else:
            a = t[:w - s]
            b = r[s:w]
        if a.size < 16:
            continue
        da = a - a.mean()
        db = b - b.mean()
        denom = np.hypot((da**2).sum(), (db**2).sum())
        val = 0.0 if denom == 0.0 else float((da @ db) / denom)
        if val > best_val:
            best_val = val
            best_shift = s
    return int(best_shift)

def _resample(line: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.float64]:
    n = int(line.size)
    if n == w:
        return line.astype(np.float64, copy=False)
    x_src = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, num=w, endpoint=False, dtype=np.float64)
    return np.interp(x_dst, x_src, line.astype(np.float64, copy=False))


# -------------------- non-live imaging --------------------

def _imaging(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData: #type:ignore

    
    """
    Map one trace and one retrace per buffer into two image rows, serpentine scan.
    Assumes channel-block DMA layout.
    """
    h: int = int(settings.height)
    w: int = int(settings.width)

    bidir: bool = bool(getattr(settings, "bidirectional", False))  # False = trace-only


    nchan: int = int(
        sum(1 for c in boards[board_index].channels if getattr(c, "enabled", False))
    )
    if nchan <= 0:
        raise ValueError("No enabled channels detected")

    rec: int = int(2 * w)
    buf_len: int = int(buf.size)
    if buf_len % nchan != 0:
        raise ValueError(
            f"Buffer length {buf_len} not divisible by channel count {nchan}"
        )
    samples_per_channel: int = buf_len // nchan
    if buf_len % nchan != 0 or samples_per_channel < w:
        raise ValueError(f"Bad buffer: len={buf_len}, nchan={nchan}, per_chan={samples_per_channel}")


    if buffer_index == 0 and board_index == 0:
        print(
            f"[CHK] rec={rec} buf_len={buf_len} nchan={nchan} samples_per_channel={samples_per_channel}"
        )
        print(f"[RAW] buf dtype={buf.dtype} shape={buf.shape} head32={buf[:32].tolist()}")
        print("[NONLIVE] active, buffer", buffer_index, "nchan", nchan, "w", w, "h", h)


    if buffer_index == 0:
        total_enabled = int(
            sum(1 for b in boards for c in b.channels if getattr(c, "enabled", False))
        )
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64) for _ in range(total_enabled)]
        )

    pairs_per_frame: int = h // 2
    pair_idx: int = int(buffer_index % max(1, pairs_per_frame))
    row0: int = 2 * pair_idx
    row1: int = row0 + 1 if row0 + 1 < h else row0

    if not bidir:
        pairs_per_frame = h  # one row per buffer
        pair_idx = int(buffer_index % max(1, pairs_per_frame))
        row0 = pair_idx
        row1 = row0  # helper will write one row


    chan_offset: int = int(
        sum(1 for b in boards[:board_index] for c in b.channels if getattr(c, "enabled", False))
    )

    #phase_frac = float(getattr(settings, "fast_motion_phase", 0.0))

    i = 0
    for c in boards[board_index].channels:
        if not c.enabled:
            continue

        # Pull one channel as float
        ch_u = _slice_channel_block(buf, nchan, i)
        mid = float(getattr(settings, "adc_midcode", 32768.0))
        ch = ( -1.0 if bool(getattr(settings, "invert_polarity", True)) else 1.0 ) * (ch_u.astype(np.float64) - mid)


        if buffer_index == 0 and board_index == 0 and i == 0:
            print(f"[POL] mid={mid} invert={getattr(settings,'invert_polarity',True)} "
                  f"ch_min={ch.min():.1f} ch_max={ch.max():.1f} ch_mean={ch.mean():.1f} "
                  f"dtype={ch.dtype} shape={ch.shape}")


        # Optional trim of head/tail samples
        head_trim = int(getattr(settings, "line_head_trim", 0))
        tail_trim = int(getattr(settings, "line_tail_trim", 0))
        if head_trim or tail_trim:
            ch = ch[head_trim : max(head_trim, ch.size - tail_trim)]

        if not bidir:
            # -------- Trace-only mode: use first half only; 1 row per buffer --------
            # Expect record ≈ 2*w; if not, just take the first half of what arrived.
            half = max(1, ch.size // 2)
            trace = _resample(ch[:half], w)
            retrace = trace  # dummy; not written when rows_per_buf=1

            # Row index advances one per buffer in trace-only mode
            row = int(buffer_index % h)

            img = data.data[chan_offset + i]
            # Write a single row; bypass per-line phase inside helper (set 0.0)
            _map_two_rows(img, row, row, trace, retrace, phase_frac=0.0, detrend=False, normalize=False)

        else:
            # -------- Bidirectional mode: split into trace + retrace; 2 rows per buffer --------
            half = ch.size // 2
            left  = ch[:half]
            right = ch[half : 2 * half]
            trace   = _resample(left,  w)
            retrace = _resample(right, w)[::-1]

            # auto-align halves
            def _estimate_shift(t, r, max_shift=64):
                wloc = int(min(t.size, r.size))
                t0 = t[:wloc] - np.median(t[:wloc])
                r0 = r[:wloc] - np.median(r[:wloc])
                m = int(min(max_shift, wloc // 4))
                best_s, best_v = 0, -1.0
                for s in range(-m, m + 1):
                    if s < 0:
                        a, b = t0[-s:wloc], r0[:wloc + s]
                    else:
                        a, b = t0[:wloc - s], r0[s:wloc]
                    if a.size < 16:
                        continue
                    da, db = a - a.mean(), b - b.mean()
                    denom = np.hypot((da**2).sum(), (db**2).sum())
                    v = 0.0 if denom == 0.0 else float((da @ db) / denom)
                    if v > best_v:
                        best_v, best_s = v, s
                return int(best_s)

            s = _estimate_shift(trace, retrace, max_shift=64)
            if s:
                trace   = np.roll(trace,   s)
                retrace = np.roll(retrace, s)
            if buffer_index == 0 and board_index == 0 and i == 0:
                print(f"[ALIGN] shift_cols={s}  w={trace.size}")

            img = data.data[chan_offset + i]
            _map_two_rows(img, row0, row1, trace, retrace, phase_frac=0.0, detrend=False, normalize=False)



        i += 1



    return cast(ProcessedData, data) #type:ignore


piezo_imaging = LiveProcessingInterface[PiezoExperimentSettings].from_function(_imaging)
