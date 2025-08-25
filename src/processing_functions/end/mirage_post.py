__all__ = ["mirage_post"]

import numpy as np
import numpy.typing as npt
from qudi.logic.mirage_logic import MirageExperimentSettings
from qudi.interface.alazar_interface import BoardInfo
from processing_functions.util.processing_defs import (
    ProcessedData,
    EndProcessingInterface,
)

"""
File for post-processing mIRage data

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
def _mirage_post(
    data: ProcessedData,
    settings: MirageExperimentSettings,
    boards: list[BoardInfo],
) -> ProcessedData:
    out = ProcessedData([])

    for i, b in enumerate(boards):
        chan_idx = 0
        num_enabled = b.count_enabled()
        for c in b.channels:
            if c.enabled:
                temp = data.data[i][chan_idx::num_enabled]
                chan_idx += 1
                p_temp = _generate_time_trace(temp, settings, label_str=f"Board {i}, Chan. {c}")
                out.data.extend(p_temp.data)
                out.labels.extend(p_temp.labels)
    return out


mirage_post = EndProcessingInterface[MirageExperimentSettings].from_function(
    _mirage_post
)


def _generate_time_trace(
    data: npt.NDArray[np.float64], settings: MirageExperimentSettings, label_str: str,
) -> ProcessedData:
    ns_per_tick = 1e9 / settings.sample_rate
    ticks_per_ir = round(settings.ir_pulse_period_us * 1e3 / ns_per_tick)
    ticks_per_wave = settings.pixel_dwell_time_us * 1e3 / ns_per_tick
    waves_per_pixel = settings.wavelengths_per_pixel
    ticks_per_pixel = waves_per_pixel * ticks_per_wave

    pts_to_average = round(ticks_per_wave / ticks_per_ir)

    # out = np.zeros(
    #     (settings.height, settings.width, settings.wavelengths_per_pixel, ticks_per_ir)
    # )

    data_list: list[npt.NDArray[np.float64]] = []
    label_list: list[str] = []

    for h in range(settings.height):
        for w in range(settings.width):
            for n in range(settings.wavelengths_per_pixel):
                trace = np.zeros(ticks_per_ir)
                init = (settings.width * h + w) * ticks_per_pixel + n * ticks_per_wave

                for i in range(pts_to_average):
                    start = round(init + i * ticks_per_ir)
                    end = round(init + (i + 1) * ticks_per_ir)
                    trace += data[start:end]

                data_list.append(trace)
                label_list.append(f"{label_str}: ({w}, {h}) at λ {n}")


    out: ProcessedData = ProcessedData(data=data_list, labels=label_list)
    return out

    