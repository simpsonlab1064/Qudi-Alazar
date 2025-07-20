__all__ = ["polarization_imaging"]

import numpy as np
import numpy.typing as npt

from qudi.logic.experiment_defs import ImagingExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.sine_time_to_pix_num import sine_time_to_pix_num
from processing_functions.util.polarization_voltage_average_image import (
    polarization_voltage_average_image,
)

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

data: ProcessedData containing the previous data from the acquisition. Note that it
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

The return should be a ProcessedData. If you intend to do imaging, it should have
the shape [data_index][data] where [data] could be 1- or 2-dimensional and 
[data_index] indicates something of meaning to you about the images (board,
channel, polarization, some combination of those, ... )

"""


def _polarization_imaging(
    data: ProcessedData,
    buf: npt.NDArray[np.int_],
    settings: ImagingExperimentSettings,
    buffer_index: int,
    board_index: int,
    boards: list[BoardInfo],
) -> ProcessedData:
    """
    Modifies data in-place, averaging for the specified number of frames. Expects
    buf to be data for a single frame
    """
    num_samples = round(len(buf) / boards[board_index].count_enabled())
    h = settings.height
    w = settings.width

    ns_per_sample = 1e9 / float(settings.sample_rate)
    total_time = ns_per_sample * num_samples

    num_enabled = boards[board_index].count_enabled()

    t = np.arange(start=ns_per_sample, stop=total_time + 1, step=ns_per_sample)
    t = sine_time_to_pix_num(
        t, w, settings.mirror_period_us * 1e3, settings.fast_mirror_phase
    )

    polarization_states = 10

    pols = np.array(range(polarization_states))
    pol_assignment = np.resize(pols, num_samples)
    offset = (num_samples * buffer_index) % polarization_states
    pol_assignment = np.roll(pol_assignment, offset)

    assignment = np.vstack([pol_assignment, t])

    total_enabled: list[int] = []
    for b in boards:
        total_enabled.append(b.count_enabled())

    # Initialize on first buffer / board and every time we've finished averaging
    if buffer_index % settings.num_frames == 0 and board_index == 0:
        data_list: list[npt.NDArray[np.float_]] = []
        for _ in range(polarization_states * np.sum(total_enabled)):
            data_list.append(np.zeros((w, h)))
        data = ProcessedData(data=data_list)

    i = 0

    for c in boards[board_index].channels:
        if c.enabled:
            temp = buf[i::num_enabled]  # note that in numpy it is start:stop:step
            temp_image = polarization_voltage_average_image(
                temp,
                assignment,
                h,
                w,
                polarization_states,
            )
            idx = (
                np.sum(
                    total_enabled[:board_index],
                    dtype=int,
                )
                * polarization_states
                + polarization_states * i
            )

            for j in range(polarization_states):
                data.data[idx + j][:, :] += temp_image[j] / settings.num_frames

            i += 1

    return data


polarization_imaging = LiveProcessingInterface[ImagingExperimentSettings].from_function(
    _polarization_imaging
)
