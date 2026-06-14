"""Data processing wrapper for Alazar digitized records."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .demodulation import digital_downconvert


class AlazarProcessor:
    """Encapsulates digital signal processing (DSP) for digitized waveforms."""

    def __init__(self, sample_rate_hz: float) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz = sample_rate_hz

    def process_decimate(
        self,
        records_volts: npt.NDArray[np.float64],
        tone_frequency_hz: float,
        reference_phase_radians: float,
        moving_average_samples: int,
        filter_type: str = "boxcar",
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
    ]:
        """Downconvert, apply digital filter, and compute the shot average."""
        baseband = digital_downconvert(
            records_volts,
            self.sample_rate_hz,
            tone_frequency_hz,
            reference_phase_radians,
        )
        filter_name = filter_type.lower()
        if filter_name == "boxcar":
            if not 1 <= moving_average_samples <= baseband.shape[1]:
                raise ValueError("moving_average_samples must fit inside the record length")
            padded = np.pad(baseband, ((0, 0), (1, 0)), mode="constant")
            cumulative = np.cumsum(padded, axis=1)
            shot_iq = (
                cumulative[:, moving_average_samples:]
                - cumulative[:, :-moving_average_samples]
            ) / moving_average_samples
        elif filter_name in ("butterworth", "butter"):
            cutoff_hz = min(max(0.75 * (2.0 * tone_frequency_hz), 75e6), 300e6)
            nyquist = self.sample_rate_hz / 2.0
            normal_cutoff = cutoff_hz / nyquist
            from scipy.signal import butter, lfilter
            b, a = butter(N=8, Wn=normal_cutoff, btype="low", analog=False)
            shot_iq = lfilter(b, a, baseband, axis=1)
        elif filter_name in ("elliptic", "ellip"):
            cutoff_hz = min(max(0.75 * (2.0 * tone_frequency_hz), 75e6), 300e6)
            nyquist = self.sample_rate_hz / 2.0
            normal_cutoff = cutoff_hz / nyquist
            from scipy.signal import ellip, lfilter
            b, a = ellip(N=5, rp=0.1, rs=40, Wn=normal_cutoff, btype="low", analog=False)
            shot_iq = lfilter(b, a, baseband, axis=1)
        elif filter_name in ("notch", "adaptive_notch"):
            theta = 2.0 * np.pi * (2.0 * tone_frequency_hz) / self.sample_rate_hz
            coef = -2.0 * np.cos(theta)
            shot_iq = np.zeros_like(baseband)
            if baseband.shape[1] >= 3:
                shot_iq[:, 1:-1] = baseband[:, 2:] + coef * baseband[:, 1:-1] + baseband[:, :-2]
                shot_iq[:, 0] = baseband[:, 0] * (2.0 + coef)
                shot_iq[:, -1] = baseband[:, -1] * (2.0 + coef)
            else:
                shot_iq = baseband * (2.0 + coef)
            shot_iq = shot_iq / (2.0 + coef)
        else:
            raise ValueError(
                f"Unknown filter_type: {filter_type}. Choose 'boxcar', 'butterworth', 'elliptic', or 'notch'."
            )

        average_iq = np.mean(shot_iq, axis=0)
        return baseband, shot_iq, average_iq

    def process_integrate(
        self,
        records_volts: npt.NDArray[np.float64],
        tone_frequency_hz: float,
        reference_phase_radians: float,
        integrate_start: int,
        integrate_stop: int,
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        np.complex128,
    ]:
        """Downconvert, integrate a specific sample window per record, and average over all records."""
        baseband = digital_downconvert(
            records_volts,
            self.sample_rate_hz,
            tone_frequency_hz,
            reference_phase_radians,
        )
        if not 0 <= integrate_start < integrate_stop <= baseband.shape[1]:
            raise ValueError("integrate window is outside the record length")

        shot_iq = np.mean(baseband[:, integrate_start:integrate_stop], axis=1)
        average_iq = np.mean(shot_iq)

        return baseband, shot_iq, np.complex128(average_iq)

    def process_multiplex_integrate(
        self,
        records_volts: npt.NDArray[np.float64],
        tone_frequencies_hz: list[float] | dict[str | int, float],
        reference_phases_radians: list[float] | dict[str | int, float] | None = None,
        integrate_start: int = 0,
        integrate_stop: int | None = None,
    ) -> dict[
        str | int | float,
        tuple[
            npt.NDArray[np.complex128],
            npt.NDArray[np.complex128],
            np.complex128,
        ],
    ]:
        """Downconvert and integrate multiple tones simultaneously from a single raw voltage trace.

        Returns a dictionary mapping each channel identifier (key in dict, index in list, or frequency float)
        to a tuple of (baseband_trace, shot_iq_points, average_iq_point).
        """
        results = {}
        
        freq_dict: dict[str | int | float, float] = {}
        if isinstance(tone_frequencies_hz, list):
            for i, freq in enumerate(tone_frequencies_hz):
                freq_dict[i] = freq
        elif isinstance(tone_frequencies_hz, dict):
            freq_dict = tone_frequencies_hz
        else:
            raise TypeError("tone_frequencies_hz must be a list or a dict")

        phase_dict: dict[str | int | float, float] = {}
        if reference_phases_radians is None:
            phase_dict = {k: 0.0 for k in freq_dict}
        elif isinstance(reference_phases_radians, list):
            for i, phase in enumerate(reference_phases_radians):
                phase_dict[i] = phase
        elif isinstance(reference_phases_radians, dict):
            phase_dict = reference_phases_radians
        else:
            raise TypeError("reference_phases_radians must be a list, dict, or None")

        for key, freq in freq_dict.items():
            phase = phase_dict.get(key, 0.0)
            baseband, shot_iq, average_iq = self.process_integrate(
                records_volts=records_volts,
                tone_frequency_hz=freq,
                reference_phase_radians=phase,
                integrate_start=integrate_start,
                integrate_stop=integrate_stop if integrate_stop is not None else records_volts.shape[1],
            )
            results[key] = (baseband, shot_iq, average_iq)

        return results

    def apply_butterworth_lpf(
        self,
        baseband_iq: npt.NDArray[np.complex128],
        cutoff_hz: float,
        order: int = 4,
    ) -> npt.NDArray[np.complex128]:
        """Apply a Butterworth low-pass filter to the complex baseband IQ traces."""
        from scipy.signal import butter, lfilter
        
        nyquist = self.sample_rate_hz / 2
        normal_cutoff = cutoff_hz / nyquist
        if not 0.0 < normal_cutoff < 1.0:
            raise ValueError(
                f"Cutoff frequency {cutoff_hz:.6g} Hz must be between DC and "
                f"Nyquist {nyquist:.6g} Hz"
            )
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return lfilter(b, a, baseband_iq, axis=1)
