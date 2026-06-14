"""Unified AWG5208 and ATS9371 experiment control."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt

from alazar import (
    ATSApi,
    AcquisitionConfig,
    AlazarProcessor,
    BoardInfo,
    TriggerConfig,
    abort_capture,
    adc_codes_to_volts,
    arm_capture,
    configure_ats9371,
    correct_interleaving_offsets,
    free_capture,
    open_ats9371,
    start_capture,
    wait_for_capture,
)
from alazar.constants import (
    CHANNEL_A,
    CHANNEL_B,
    MIN_SAMPLES_PER_RECORD,
    SAMPLES_PER_RECORD_ALIGNMENT,
)
from awg5200 import AWG5208


def seconds_to_samples(duration_s: float, sample_rate_hz: float) -> int:
    """Convert seconds to the nearest integer sample count."""
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return int(round(duration_s * sample_rate_hz))


def samples_to_seconds(number_of_samples: int, sample_rate_hz: float) -> float:
    """Convert a sample count to seconds."""
    if number_of_samples < 0:
        raise ValueError("number_of_samples cannot be negative")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return number_of_samples / sample_rate_hz


def records_per_buffer_for(num_averages: int, maximum: int = 100) -> int:
    """Choose the largest small DMA buffer that divides num_averages."""
    if num_averages < 1:
        raise ValueError("num_averages must be positive")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    for candidate in range(min(num_averages, maximum), 0, -1):
        if num_averages % candidate == 0:
            return candidate
    raise RuntimeError("No records_per_buffer divisor found")


def normalize_adc_channel(channel: str | int) -> int:
    """Map CHA/CHB or zero-based 0/1 to an ATS channel constant."""
    if isinstance(channel, str):
        name = channel.strip().upper()
        if name == "CHA":
            return CHANNEL_A
        if name == "CHB":
            return CHANNEL_B
    elif isinstance(channel, int) and not isinstance(channel, bool):
        if channel == 0:
            return CHANNEL_A
        if channel == 1:
            return CHANNEL_B
    raise ValueError("adc_channel must be 'CHA', 'CHB', 0, or 1")


class AWGAlazar:
    """Coordinate AWG playback, triggered acquisition, and IQ averaging."""

    def __init__(
        self,
        awg: AWG5208,
        ats_api: ATSApi,
        ats_board: BoardInfo,
        *,
        awg_sample_rate_hz: float,
        alazar_sample_rate_hz: float,
        tone_frequency_hz: float,
        trigger_delay_s: float,
        acquire_window_ns: float,
        integrate_window_ns: tuple[float, float],
        adc_channel: str | int = "CHA",
        moving_average_time_s: float = 20e-9,
        reference_phase_radians: float = 0.0,
        trigger_level: int = 140,
        input_range_volts: float = 0.4,
        timeout_ms: int = 5000,
        dma_buffer_count: int = 4,
        maximum_records_per_buffer: int = 100,
        use_external_10mhz_reference: bool = True,
        baseline_time_s: float | None = None,
    ) -> None:
        self.awg = awg
        self.ats_api = ats_api
        self.ats_board = ats_board
        self.awg_sample_rate_hz = float(awg_sample_rate_hz)
        self.alazar_sample_rate_hz = float(alazar_sample_rate_hz)
        self.tone_frequency_hz = float(tone_frequency_hz)
        self.trigger_delay_s = float(trigger_delay_s)
        self.acquire_window_ns = float(acquire_window_ns)
        self.integrate_window_ns = (
            float(integrate_window_ns[0]),
            float(integrate_window_ns[1]),
        )
        self.adc_channel = normalize_adc_channel(adc_channel)
        self.moving_average_time_s = float(moving_average_time_s)
        self.reference_phase_radians = float(reference_phase_radians)
        self.trigger_level = int(trigger_level)
        self.input_range_volts = float(input_range_volts)
        self.timeout_ms = int(timeout_ms)
        self.dma_buffer_count = int(dma_buffer_count)
        self.maximum_records_per_buffer = int(maximum_records_per_buffer)
        self.use_external_10mhz_reference = bool(use_external_10mhz_reference)
        self.baseline_time_s = baseline_time_s

        self.last_raw_codes: npt.NDArray[np.uint16] | None = None
        self.last_records_volts: npt.NDArray[np.float64] | None = None
        self.last_downconverted_iq: (
            npt.NDArray[np.complex128] | None
        ) = None
        self.last_shot_iq: npt.NDArray[np.complex128] | None = None
        self.last_time_s: npt.NDArray[np.float64] | None = None
        self.last_sequence_records_volts: (
            npt.NDArray[np.float64] | None
        ) = None
        self.last_sequence_shot_iq: (
            npt.NDArray[np.complex128] | None
        ) = None

        self._validate_settings()
        self.processor = AlazarProcessor(self.alazar_sample_rate_hz)

    @classmethod
    def connect(
        cls,
        awg_resource: str,
        *,
        awg_sample_rate_hz: float,
        alazar_sample_rate_hz: float = 1e9,
        tone_frequency_hz: float,
        trigger_delay_s: float,
        acquire_window_ns: float,
        integrate_window_ns: tuple[float, float],
        awg_timeout_ms: int = 60_000,
        ats_system_id: int = 1,
        ats_board_id: int = 1,
        **settings: Any,
    ) -> "AWGAlazar":
        """Connect both instruments and apply their clock/trigger settings."""
        awg = AWG5208.connect(awg_resource, timeout_ms=awg_timeout_ms)
        try:
            ats_api = ATSApi()
            ats_board = open_ats9371(ats_api, ats_system_id, ats_board_id)
            experiment = cls(
                awg,
                ats_api,
                ats_board,
                awg_sample_rate_hz=awg_sample_rate_hz,
                alazar_sample_rate_hz=alazar_sample_rate_hz,
                tone_frequency_hz=tone_frequency_hz,
                trigger_delay_s=trigger_delay_s,
                acquire_window_ns=acquire_window_ns,
                integrate_window_ns=integrate_window_ns,
                **settings,
            )
            experiment.configure()
            return experiment
        except BaseException:
            awg.close()
            raise

    def _validate_settings(self) -> None:
        if not 1.49e3 <= self.awg_sample_rate_hz <= 2.5e9:
            raise ValueError(
                "awg_sample_rate_hz must be between 1.49 kSa/s and 2.5 GSa/s"
            )
        if self.alazar_sample_rate_hz != 1e9:
            raise ValueError("ATS9371 acquisition is currently fixed to 1 GS/s")
        if self.input_range_volts != 0.4:
            raise ValueError(
                "The current ATS9371 driver configures a fixed +/-400 mV range"
            )
        if not 0 <= self.tone_frequency_hz < self.alazar_sample_rate_hz / 2:
            raise ValueError("tone_frequency_hz must be between DC and Nyquist")
        if self.trigger_delay_s < 0:
            raise ValueError("trigger_delay_s cannot be negative")
        if self.acquire_window_ns <= 0:
            raise ValueError("acquire_window_ns must be positive")
        integrate_start_ns, integrate_stop_ns = self.integrate_window_ns
        if not 0 <= integrate_start_ns < integrate_stop_ns:
            raise ValueError(
                "integrate_window_ns must satisfy 0 <= start < stop"
            )
        if integrate_stop_ns > self.acquire_window_ns:
            raise ValueError(
                "integrate_window_ns must fit inside acquire_window_ns"
            )
        if not 0 < self.moving_average_time_s * 1e9 <= self.acquire_window_ns:
            raise ValueError(
                "moving_average_time_s must fit inside acquire_window_ns"
            )
        if self.baseline_time_s is not None:
            if not 0 < self.baseline_time_s * 1e9 <= self.acquire_window_ns:
                raise ValueError(
                    "baseline_time_s must fit inside acquire_window_ns"
                )

    @property
    def trigger_delay_samples(self) -> int:
        return self.ns2cycles(self.trigger_delay_s * 1e9, inst="adc")

    @property
    def adc_channel_name(self) -> str:
        return "CHA" if self.adc_channel == CHANNEL_A else "CHB"

    @property
    def adc_lsb_volts(self) -> float:
        return self.input_range_volts / (
            1 << (self.ats_board.bits_per_sample - 1)
        )

    @property
    def integrate_window_cycles(self) -> tuple[int, int]:
        start_ns, stop_ns = self.integrate_window_ns
        return (
            self.ns2cycles(start_ns, inst="adc"),
            self.ns2cycles(stop_ns, inst="adc"),
        )

    @property
    def integrate_samples(self) -> int:
        start, stop = self.integrate_window_cycles
        return stop - start

    @property
    def acquire_window_cycles(self) -> int:
        requested = self.ns2cycles(self.acquire_window_ns, inst="adc")
        alignment = SAMPLES_PER_RECORD_ALIGNMENT
        aligned = ((requested + alignment - 1) // alignment) * alignment
        return max(aligned, MIN_SAMPLES_PER_RECORD)

    @property
    def moving_average_samples(self) -> int:
        samples = self.ns2cycles(
            self.moving_average_time_s * 1e9,
            inst="adc",
        )
        if samples < 1:
            raise ValueError("moving_average_time_s is shorter than one sample")
        return samples

    def _sample_rate_for(self, inst: str) -> float:
        instrument = inst.lower()
        if instrument == "dac":
            return self.awg_sample_rate_hz
        if instrument == "adc":
            return self.alazar_sample_rate_hz
        raise ValueError("inst must be 'dac' for AWG or 'adc' for Alazar")

    def ns2cycles(self, duration_ns: float, inst: str = "dac") -> int:
        """Convert nanoseconds to AWG DAC or Alazar ADC sample cycles."""
        return seconds_to_samples(
            duration_ns * 1e-9,
            self._sample_rate_for(inst),
        )

    def cycles2ns(self, cycles: int, inst: str = "dac") -> float:
        """Convert AWG DAC or Alazar ADC sample cycles to nanoseconds."""
        return samples_to_seconds(
            cycles,
            self._sample_rate_for(inst),
        ) * 1e9

    def configure(self) -> None:
        """Configure clocks, AWG mode/sample rate, and the ATS trigger delay."""
        self.awg.set_awg_mode()
        if self.use_external_10mhz_reference:
            self.awg.use_external_10mhz_reference()
        self.awg.set_sample_rate(self.awg_sample_rate_hz)
        configure_ats9371(
            self.ats_api,
            self.ats_board,
            TriggerConfig(
                level=self.trigger_level,
                delay_samples=self.trigger_delay_samples,
                timeout_ticks=0,
            ),
            use_external_10mhz_reference=self.use_external_10mhz_reference,
            channel=self.adc_channel,
        )

    def _acquisition_config(self, n_average: int) -> AcquisitionConfig:
        num_averages = int(n_average)
        if num_averages < 1:
            raise ValueError("n_average must be positive")
        return AcquisitionConfig(
            sample_rate_hz=self.alazar_sample_rate_hz,
            tone_frequency_hz=self.tone_frequency_hz,
            samples_per_record=self.acquire_window_cycles,
            num_averages=num_averages,
            records_per_buffer=records_per_buffer_for(
                num_averages,
                self.maximum_records_per_buffer,
            ),
            dma_buffer_count=self.dma_buffer_count,
            input_range_volts=self.input_range_volts,
            timeout_ms=self.timeout_ms,
            channel=self.adc_channel,
        )

    def _capture_records(
        self,
        n_average: int,
    ) -> npt.NDArray[np.float64]:
        config = self._acquisition_config(n_average=n_average)
        self.awg.stop()
        session = arm_capture(self.ats_api, self.ats_board, config)
        try:
            start_capture(self.ats_api, session)
            self.awg.run(wait_until_ready=False)
            raw_codes = wait_for_capture(
                self.ats_api,
                session,
                config.timeout_ms,
            )
        finally:
            with suppress(Exception):
                self.awg.stop()
            with suppress(Exception):
                abort_capture(self.ats_api, session)
            free_capture(session)

        records = adc_codes_to_volts(
            raw_codes,
            self.ats_board.bits_per_sample,
            config.input_range_volts,
        )
        if self.baseline_time_s is not None:
            records = correct_interleaving_offsets(
                records,
                stop_sample=self.ns2cycles(
                    self.baseline_time_s * 1e9,
                    inst="adc",
                ),
                period=2,
            )

        self.last_raw_codes = raw_codes
        self.last_records_volts = records
        return records

    def acquire_decimate(
        self,
        n_average: int,
        filter_type: str = "boxcar",
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
    ]:
        """Acquire and return a shot-averaged, time-resolved IQ waveform."""
        records = self._capture_records(n_average=n_average)
        baseband, shot_iq, average_iq = self.processor.process_decimate(
            records_volts=records,
            tone_frequency_hz=self.tone_frequency_hz,
            reference_phase_radians=self.reference_phase_radians,
            moving_average_samples=self.moving_average_samples,
            filter_type=filter_type,
        )
        time_s = (
            np.arange(average_iq.size, dtype=np.float64)
            / self.alazar_sample_rate_hz
        )

        self.last_downconverted_iq = baseband
        self.last_shot_iq = shot_iq
        self.last_time_s = time_s
        return time_s, average_iq

    def acquire(
        self,
        n_average: int,
    ) -> tuple[
        np.complex128,
        npt.NDArray[np.complex128],
    ]:
        """Return one averaged IQ point and every downconverted shot trace."""
        records = self._capture_records(n_average=n_average)
        integrate_start, integrate_stop = self.integrate_window_cycles
        baseband, shot_iq, average_iq = self.processor.process_integrate(
            records_volts=records,
            tone_frequency_hz=self.tone_frequency_hz,
            reference_phase_radians=self.reference_phase_radians,
            integrate_start=integrate_start,
            integrate_stop=integrate_stop,
        )

        self.last_downconverted_iq = baseband
        self.last_shot_iq = shot_iq
        self.last_time_s = (
            np.arange(baseband.shape[1], dtype=np.float64)
            / self.alazar_sample_rate_hz
        )
        return average_iq, baseband

    def acquire_sequence_traces(
        self,
        number_of_steps: int,
        number_of_averages: int,
        filter_type: str = "boxcar",
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
    ]:
        """Acquire an interleaved sequence and average matching traces."""
        steps = int(number_of_steps)
        averages = int(number_of_averages)
        if steps < 1 or averages < 1:
            raise ValueError("number_of_steps and number_of_averages must be positive")

        records = self._capture_records(n_average=steps * averages)
        sequence_records = records.reshape(
            averages,
            steps,
            records.shape[1],
        )
        average_records = np.mean(sequence_records, axis=0)

        baseband, shot_iq, _ = self.processor.process_decimate(
            records_volts=records,
            tone_frequency_hz=self.tone_frequency_hz,
            reference_phase_radians=self.reference_phase_radians,
            moving_average_samples=self.moving_average_samples,
            filter_type=filter_type,
        )
        sequence_shot_iq = shot_iq.reshape(
            averages,
            steps,
            shot_iq.shape[1],
        )
        average_iq = np.mean(sequence_shot_iq, axis=0)
        raw_time_s = (
            np.arange(records.shape[1], dtype=np.float64)
            / self.alazar_sample_rate_hz
        )
        iq_time_s = (
            np.arange(shot_iq.shape[1], dtype=np.float64)
            / self.alazar_sample_rate_hz
        )

        self.last_downconverted_iq = baseband
        self.last_shot_iq = shot_iq
        self.last_time_s = iq_time_s
        self.last_sequence_records_volts = sequence_records
        self.last_sequence_shot_iq = sequence_shot_iq
        return raw_time_s, average_records, iq_time_s, average_iq

    def capture_diagnostics(self) -> dict[str, float | int | str]:
        """Summarize the most recent raw acquisition without modifying it."""
        if self.last_raw_codes is None or self.last_records_volts is None:
            raise RuntimeError("Run acquire() or acquire_decimate() first")
        average = np.mean(self.last_records_volts, axis=0)
        return {
            "adc_channel": self.adc_channel_name,
            "adc_bits": self.ats_board.bits_per_sample,
            "adc_lsb_mv": self.adc_lsb_volts * 1e3,
            "raw_code_min": int(np.min(self.last_raw_codes)),
            "raw_code_max": int(np.max(self.last_raw_codes)),
            "mean_offset_mv": float(np.mean(self.last_records_volts) * 1e3),
            "average_peak_to_peak_mv": float(np.ptp(average) * 1e3),
            "shot_noise_std_mv": float(
                np.std(self.last_records_volts - average[None, :]) * 1e3
            ),
        }

    def close(self) -> None:
        self.awg.close()

    def __enter__(self) -> "AWGAlazar":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
