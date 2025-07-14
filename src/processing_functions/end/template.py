import numpy as np

from qudi.logic.base_alazar_logic import BaseExperimentSettings
from qudi.interface.alazar_interface import BoardInfo

"""
This file contains a template for end-processing functions.

End functions _must_ conform to the function signature in this file and they
must be in a file that has the same name as the function being called (e.g.
template() is called from template.py)

You are welcome to put additional / helper functions in this file or wherever
you would like (as long as you use full module paths to access them)

The arguments are as follows:

data: An array containing all of the data from the acquisition. It is organized
      like [boardId, data]. If multiple channels are enabled, the data is
      interleaved. If you are using a live-processing function along with an end
      processing function, you are responsible for making sure the end function
      knows what the layout of the data is.
settings: All of the experimental settings for your measurement (image w/h,
          number of frames, ...). Probably you should type hint the correct
          ExperimentSettings for what you're doing.
boards: List of boards in the system -- for determining measurement type / if
        a given channel is enabled

The return should be an np.ndarray. If you intend to do imaging, it should have
the shape [data_index][data] where [data] could be 1- or 2-dimensional

"""


def template(
    data: np.ndarray, settings: BaseExperimentSettings, boards: list[BoardInfo]
) -> np.ndarray:
    out = np.zeros((512 * 512,))
    for i, b in enumerate(boards):
        d = data[i]
        c_idx = 0
        for _, c in enumerate(b.channels):
            if c.enabled:
                temp = d[c_idx : b.count_enabled() * 512 * 512 : b.count_enabled()]
                out = out + temp
                c_idx += 1
    np.reshape(out, (512, 512))
    return out
