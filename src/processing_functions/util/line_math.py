# processing_functions/util/line_math.py
from __future__ import annotations

def line_index_and_serpentine(
    buffer_index: int,
    height: int,
    frames: int,
) -> tuple[int, bool]:
    """
    Compute current line index and serpentine flag for bidirectional fast axis.
    """
    if height <= 0 or frames <= 0:
        raise ValueError("height and frames must be positive")
    line = (buffer_index // frames) % height
    serpentine = (line & 1) == 1
    return line, serpentine
