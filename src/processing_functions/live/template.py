__all__ = ["template"]

import numpy as np

from qudi.logic.base_alazar_logic import BaseExperimentSettings
from qudi.interface.alazar_interface import BoardInfo

"""
This file contains a template for live-processing functions.

Live functions _must_ conform to the function signature in this file and they
must be in a file that has the same name as the function being called (e.g.
template() is called from template.py). They also should specify __all__[]
at the top of the file (like this one). This is technically optional, but is 
good practice

You are welcome to put additional / helper functions in this file or wherever
you would like (as long as you use full module paths to access them)

The arguments are as follows:

data: An array containing the previous data from the acquisition. Note that it
      will be empty on the first buffer as there is no data yet
buf: Array containing the most recently acquired buffer
settings: All of the experimental settings for your measurement (image w/h,
          number of frames, ...). Probably you should type hint the correct
          ExperimentSettings for what you're doing.
buffer_index: What buffer number are we on? (this increments once all boards have
                supplied one buffer, so it is equivalent to frame number if you
                are doing imaging)
board_index: Which board is this buffer from?
boards: List of boards in the system -- for determining measurement type / if
        a given channel is enabled

"""


def template(
    data: np.ndarray,
    buf: np.ndarray,
    settings: BaseExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> np.ndarray:
    if buffer_index == 0 and len(data) == 0:
        out = np.zeros((len(boards), len(buf)))
        out[board_index, :] = buf
        return out
    else:
        data[board_index, :] += buf
        return data
