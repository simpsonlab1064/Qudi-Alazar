# processing_functions/util/records_to_row.py
from __future__ import annotations
import numpy as np
import numpy.typing as npt

def mean_per_record(
    signal: npt.NDArray[np.float64] | npt.NDArray[np.int_],
    samples_per_record: int,
    records_per_buffer: int,
) -> npt.NDArray[np.float64]:
    """
    Convert a 1D channel trace into one averaged value per record.
    Returns a vector of length records_per_buffer.
    """
    if samples_per_record <= 0:
        raise ValueError("samples_per_record must be positive")
    vec = np.asarray(signal, dtype=np.float64).ravel()
    usable = (vec.size // samples_per_record) * samples_per_record
    if usable == 0:
        return np.zeros(records_per_buffer, dtype=np.float64)
    out = vec[:usable].reshape(-1, samples_per_record).mean(axis=1)
    # safety trim or pad to the advertised record count
    out = out[:records_per_buffer]
    if out.size < records_per_buffer:
        out = np.pad(out, (0, records_per_buffer - out.size), constant_values=0)
    return out
