__all__ = ["template"]

import numpy as np
from qudi.logic.base_alazar_logic import BaseExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    EndProcessingInterface,
)

"""
This file contains a template for end-processing functions.

End functions _must_ conform to the function signature in this file and they
must be in a file that has the same name as the function being called (e.g.
template() is called from template.py)

You are welcome to put additional / helper functions in this file or wherever
you would like (as long as you use full module paths to access them)

The arguments are as follows:

data: ProcessedData from the utility folder definition
settings: All of the experimental settings for your measurement (image w/h,
          number of frames, ...). Probably you should type hint the correct
          ExperimentSettings for what you're doing.
boards: List of boards in the system -- for determining measurement type / if
        a given channel is enabled

The return should be a ProcessedData object

"""


# Functions must use the variable names here (data, settings, boards)
def _template(
    data: ProcessedData,
    settings: BaseExperimentSettings,
    boards: list[BoardInfo],
) -> ProcessedData:
    out = np.zeros((512 * 512,))
    for i, b in enumerate(boards):
        d = data.data[i]
        c_idx = 0
        for _, c in enumerate(b.channels):
            if c.enabled:
                temp = d[c_idx : b.count_enabled() * 512 * 512 : b.count_enabled()]
                out = out + temp
                c_idx += 1
    np.reshape(out, (512, 512))

    return ProcessedData(data=[out])


template = EndProcessingInterface[BaseExperimentSettings].from_function(_template)
