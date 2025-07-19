__all__ = ["hybrid_image"]

import numpy as np
import numpy.typing as npt


def hybrid_image(
    data: npt.NDArray[np.int_],
    avg_image: npt.NDArray[np.float_],
    pc_image: npt.NDArray[np.float_],
    lambda_threshold: int,
    voltage_range: int,
) -> npt.NDArray[np.float_]:
    adc_zero, volts_per_photon = _calibrate_volts_per_photon(
        data, avg_image, pc_image, voltage_range
    )

    image = -1 * (avg_image - adc_zero) / volts_per_photon
    image[pc_image < lambda_threshold] = pc_image[pc_image < lambda_threshold]

    return image


def _calibrate_volts_per_photon(
    data: npt.NDArray[np.int_],
    avg_image: npt.NDArray[np.float_],
    pc_image: npt.NDArray[np.float_],
    voltage_range: int,
) -> tuple[int, float]:
    """
    Returns (adc_zero, volts_per_photon)
    """
    _, hist = np.histogram(
        data, np.arange(2**15 - voltage_range, 2**15 + voltage_range, 1)
    )
    johnson_noise_mode = np.max(hist)
    adc_zero: int = johnson_noise_mode + 2**15 - voltage_range - 1
    guess_volts_per_photon_image = np.abs((avg_image - adc_zero) / pc_image)
    mean_guess_volts_per_photon_image = np.mean(guess_volts_per_photon_image[:])
    hist_range = np.linspace(
        0.3 * mean_guess_volts_per_photon_image,
        1.5 * mean_guess_volts_per_photon_image,
        1000,
    )

    _, hist = np.histogram(
        guess_volts_per_photon_image[np.isfinite(guess_volts_per_photon_image)],
        hist_range,
    )

    volts_per_photon_mode: int = np.max(hist)

    return adc_zero, hist_range[volts_per_photon_mode]
