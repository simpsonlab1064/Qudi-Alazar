__all__ = ["piezo_imaging_live"]

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
    # Interleaved layout: A0, B0, A1, B1, ...
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
    transpose: bool = False,
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

    t0 = _prep(trace)
    t1 = _prep(retrace)

    if not transpose:
        img[row0, : t0.size] = t0
        if row1 != row0:
            img[row1, : t1.size] = t1
    else:
        img[: t0.size, row0] = t0
        if row1 != row0:
            img[: t1.size, row1] = t1


    # optional phase roll applied to each line
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


# -------------------- live imaging --------------------

def _imaging_live(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: PiezoExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    """
    Map one trace and one retrace per buffer into two image rows, serpentine scan.
    Assumes channel-block DMA layout.
    """
    h: int = int(settings.height)
    w: int = int(settings.width)

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
    if samples_per_channel != rec:
        raise ValueError(
            f"Unexpected samples per channel {samples_per_channel}; expected {rec} "
            f"(set records_per_buffer=1 and samples_per_record=2*width)"
        )

    if buffer_index == 0 and board_index == 0:
        print(
            f"[CHK] rec={rec} buf_len={buf_len} nchan={nchan} samples_per_channel={samples_per_channel}"
        )

    if buffer_index == 0:
        total_enabled = int(sum(b.count_enabled() for b in boards))
        data = ProcessedData(
            data=[np.zeros((h, w), dtype=np.float64) for _ in range(total_enabled)]
        )

    pairs_per_frame: int = h // 2
    pair_idx: int = int(buffer_index % max(1, pairs_per_frame))
    row0: int = 2 * pair_idx
    row1: int = row0 + 1 if row0 + 1 < h else row0

    chan_offset: int = int(sum(b.count_enabled() for b in boards[:board_index]))

    #phase_frac = float(getattr(settings, "fast_motion_phase", 0.0))

    i = 0
    for c in boards[board_index].channels:
        if not c.enabled:
            continue

        # deinterleave one channel from a sample interleaved buffer
        ch_u = _slice_channel_block(buf, nchan, i)
        mid = float(getattr(settings, "adc_midcode", 32768.0))
        ch = ( -1.0 if bool(getattr(settings, "invert_polarity", True)) else 1.0 ) * (ch_u.astype(np.float64) - mid)

        if buffer_index == 0 and board_index == 0 and i == 0:
            print(f"[POL] mid={mid} invert={getattr(settings,'invert_polarity',True)} "
                  f"ch_min={ch.min():.1f} ch_max={ch.max():.1f} ch_mean={ch.mean():.1f} "
                  f"dtype={ch.dtype} shape={ch.shape}")


        rec_len = int(ch.size)

        # optional trimming of bad head or tail samples
        head_trim = int(getattr(settings, "line_head_trim", 0))
        tail_trim = int(getattr(settings, "line_tail_trim", 0))
        trimmed = ch[head_trim : max(head_trim, rec_len - tail_trim)]
        rec_len = int(trimmed.size)

        # split the record into two halves
        half_len = rec_len // 2
        left = trimmed[:half_len]
        right = trimmed[half_len : 2 * half_len]

        # enforce equal halves
        if left.size != right.size:
            m = min(left.size, right.size)
            left = left[:m]
            right = right[:m]
            half_len = m

        # resample utility
        def _rs(x: npt.NDArray[np.float64], wdst: int) -> npt.NDArray[np.float64]:
            n = int(x.size)
            if n == wdst:
                return x.astype(np.float64, copy=False)
            xs = np.linspace(0.0, 1.0, num=n,    endpoint=False, dtype=np.float64)
            xd = np.linspace(0.0, 1.0, num=wdst, endpoint=False, dtype=np.float64)
            return np.interp(xd, xs, x.astype(np.float64, copy=False))

        bidir = bool(getattr(settings, "bidirectional", True))

        if buffer_index == 0 and i == 0 and board_index == 0:
            print(f"[PROC] bidirectional={bidir}")


        if not bidir:
            # ---------- TRACE-ONLY: one row per buffer ----------
            trace = _rs(left, w)
            # row index advances by one per buffer in trace-only mode
            row = int(buffer_index % h)

            # diagnostics: per-line stats and correlation with previous line
            if not hasattr(settings, "last_line_stats"):
                setattr(settings, "last_line_stats", None)
            prev = getattr(settings, "last_line_stats")

            m = float(trace.mean())
            s = float(trace.std())
            if prev is not None:
                prev_line = prev["line"]
                # length guard
                n = int(min(prev_line.size, trace.size))
                a = prev_line[:n] - prev_line[:n].mean()
                b = trace[:n] - trace[:n].mean()
                denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
                corr = 0.0 if denom == 0.0 else float((a * b).sum() / denom)
            else:
                corr = 0.0

            if buffer_index < 8 and i == 0 and board_index == 0:
                print(f"[LINE] buf={buffer_index} row={int(buffer_index % h)} mean={m:.2f} std={s:.2f} corr_prev={corr:.3f}")

            setattr(settings, "last_line_stats", {"line": trace.copy()})


            img = data.data[chan_offset + i]
            _map_two_rows(
                img,
                row0,
                row1,
                trace,
                retrace,
                phase_frac=0.0,
                detrend=False,
                normalize=False,
                transpose=bool(getattr(settings, "transpose_image", False)),
            )

        else:
            # ---------- BIDIRECTIONAL: trace + retrace; two rows per buffer ----------
            trace = _rs(left,  w)
            retrace_raw = _rs(right, w)

            # choose retrace direction based on setting
            rev = bool(getattr(settings, "reverse_retrace", True))
            retrace = retrace_raw[::-1] if rev else retrace_raw

            # center-window fractional alignment applied to RETRACE ONLY
            def _est_shift_center_frac(t: npt.NDArray[np.float64],
                                       r: npt.NDArray[np.float64],
                                       max_shift: int = 64) -> float:
                wloc = int(min(t.size, r.size))
                if wloc <= 0:
                    return 0.0
                i0 = int(0.20 * wloc)
                i1 = int(0.80 * wloc)

                def _filt(xx: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                    if xx.size < 9:
                        return xx
                    k = np.ones(9, dtype=np.float64) / 9.0
                    return np.convolve(xx, k, mode="same")

                t0 = _filt(t[i0:i1].astype(np.float64, copy=False))
                r0 = _filt(r[i0:i1].astype(np.float64, copy=False))
                t0 -= float(np.median(t0))
                r0 -= float(np.median(r0))

                m = int(min(max_shift, t0.size // 4))
                best_s = 0
                best_v = -1.0
                for sft in range(-m, m + 1):
                    a = t0[-sft:] if sft < 0 else t0[:t0.size - sft]
                    b = r0[:r0.size + sft] if sft < 0 else r0[sft:]
                    n = int(min(a.size, b.size))
                    if n < 16:
                        continue
                    aa = a[:n] - a[:n].mean()
                    bb = b[:n] - b[:n].mean()
                    denom = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
                    v = 0.0 if denom == 0.0 else float((aa * bb).sum() / denom)
                    if v > best_v:
                        best_v = v
                        best_s = sft

                # quadratic refinement
                def score(sft: int) -> float:
                    a = t0[-sft:] if sft < 0 else t0[:t0.size - sft]
                    b = r0[:r0.size + sft] if sft < 0 else r0[sft:]
                    n = int(min(a.size, b.size))
                    if n < 16:
                        return -1.0
                    aa = a[:n] - a[:n].mean()
                    bb = b[:n] - b[:n].mean()
                    denom = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
                    return 0.0 if denom == 0.0 else float((aa * bb).sum() / denom)

                s0 = best_s
                s_1, s1 = s0 - 1, s0 + 1
                v_1, v0, v1 = score(s_1), score(s0), score(s1)
                denom_q = (v_1 - 2.0 * v0 + v1)
                frac = 0.0 if denom_q == 0.0 else 0.5 * (v_1 - v1) / denom_q
                return float(s0 + np.clip(frac, -0.5, 0.5))

            s_now = _est_shift_center_frac(trace, retrace, max_shift=64)
            last = float(getattr(settings, "last_align_shift", 0.0))
            alpha = 0.9
            s = alpha * last + (1.0 - alpha) * float(s_now)
            setattr(settings, "last_align_shift", s)

            if abs(s) > 1e-6:
                idx = np.arange(retrace.size, dtype=np.float64)
                idx = (idx - s) % retrace.size
                retrace = np.interp(idx, np.arange(retrace.size, dtype=np.float64), retrace)

            if buffer_index == 0 and i == 0 and board_index == 0:
                print(f"[ALIGN] est={s_now:.2f} smooth={s:.2f} rev={rev} w={trace.size}")

            img = data.data[chan_offset + i]
            _map_two_rows(
                img,
                row0,
                row1,
                trace,
                retrace,
                phase_frac=0.0,
                detrend=False,
                normalize=False,
            )






        i += 1


    return data


piezo_imaging_live = LiveProcessingInterface[PiezoExperimentSettings].from_function(
    _imaging_live
)
