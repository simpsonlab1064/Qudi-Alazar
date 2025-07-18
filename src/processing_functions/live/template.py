__all__ = ["template"]

import numpy as np
import numpy.typing as npt
from qudi.logic.base_alazar_logic import BaseExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    LiveProcessingInterface,
)


"""
This file contains a template for live-processing functions.

Live functions _must_ conform to the function signature in this file and they
must be in a file that has the same name as the function being called (e.g.
template() is called from template.py).

You are welcome to put additional / helper functions in this file or wherever
you would like (as long as you use full module paths to access them)

The arguments are as follows:

data: ProcessedData object -- note that it will have no data on first run
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

The return should be an ProcessedData object. If you intend to do imaging, it's
data should have the shape [data_index][data] where [data] could be 1- or 2-dimensional

Ideally you will export the function using the LiveProcessingInterface.from_function(func)
interface to get compile/run time checks of the type and return signatures
"""


# Functions must use the argument names from here: data, buf, settings, buffer_index,
# board_index, boards
def _template(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],  # or could be np.float_
    settings: BaseExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    if buffer_index == 0 and len(data.data) == 0:
        return ProcessedData(data=[buf])
    else:
        data.data[board_index] += buf
        return data


template = LiveProcessingInterface[BaseExperimentSettings].from_function(_template)
