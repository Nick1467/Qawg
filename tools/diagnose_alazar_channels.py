from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAWG.awg_alazar import AWGAlazar


AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
SAMPLE_RATE_HZ = 1e9
TONE_FREQUENCY_HZ = 50e6


def channel_statistics(records: np.ndarray) -> dict[str, float]:
    average = np.mean(records, axis=0)
    centered = average - np.mean(average)
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(average.size, 1.0 / SAMPLE_RATE_HZ)
    tone_index = int(np.argmin(np.abs(frequencies - TONE_FREQUENCY_HZ)))
    return {
        "mean_mv": float(np.mean(records) * 1e3),
        "average_p2p_mv": float(np.ptp(average) * 1e3),
        "shot_std_mv": float(np.std(records - average[None, :]) * 1e3),
        "tone_fft": float(spectrum[tone_index]),
    }


for adc_channel in ("CHA", "CHB"):
    experiment = AWGAlazar.connect(
        AWG_RESOURCE,
        awg_sample_rate_hz=2.5e9,
        alazar_sample_rate_hz=SAMPLE_RATE_HZ,
        tone_frequency_hz=TONE_FREQUENCY_HZ,
        trigger_delay_s=0.0,
        acquire_window_ns=1200,
        integrate_window_ns=(100, 1100),
        adc_channel=adc_channel,
        moving_average_time_s=20e-9,
        timeout_ms=60_000,
        use_external_10mhz_reference=True,
    )
    try:
        _, _ = experiment.acquire(100)
        raw_codes = experiment.last_raw_codes
        records = experiment.last_records_volts
        stats = channel_statistics(records)
        print(f"\n{adc_channel}")
        print(f"  normalized channel: {experiment.adc_channel_name}")
        print(f"  shape: {records.shape}")
        print(f"  raw code range: {raw_codes.min()} .. {raw_codes.max()}")
        for name, value in stats.items():
            print(f"  {name}: {value:.6g}")
    finally:
        experiment.close()
