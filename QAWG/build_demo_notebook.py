from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "scqenv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

notebook["cells"] = [
    markdown(
        """
# QAWG experiment compiler demo

This notebook demonstrates the unified host-side compiler without requiring
hardware for the preview sections.

Included examples:

1. Pulse-probe spectroscopy frequency sequence.
2. Power Rabi gain sequence.
3. T1 delay sequence.
4. Single-shot ground/excited sequence.
5. Optional AWG5208 + ATS9371 acquisition using `n_average`.
"""
    ),
    code(
        """
import numpy as np
import matplotlib.pyplot as plt

from QAWG import (
    MHz,
    PowerRabiProgram,
    PulseProbeSpectroscopyProgram,
    SingleShotProgram,
    T1Program,
    ns,
    us,
)
"""
    ),
    markdown("## Shared configuration"),
    code(
        """
SAMPLE_RATE_HZ = 2.5e9

base_cfg = {
    "qubit_ch": 4,
    "res_ch": 3,
    "marker_ch": 1,
    "adc_channel": "CHA",
    "f_ge": 100 * MHz,
    "f_res": 50 * MHz,
    "res_len": 1 * us,
    "res_gain": 0.02,
    "ro_len": 1 * us,
}
"""
    ),
    markdown("## Pulse-probe spectroscopy"),
    code(
        """
spectroscopy_cfg = {
    **base_cfg,
    "frequency_start": 80 * MHz,
    "frequency_stop": 120 * MHz,
    "steps": 21,
    "probe_len": 500 * ns,
    "qubit_gain": 0.02,
    "trig_time": 0,
}

spectroscopy = PulseProbeSpectroscopyProgram(spectroscopy_cfg)
spectroscopy_compiled = spectroscopy.compile(
    sample_rate_hz=SAMPLE_RATE_HZ,
)

print("Sequence steps:", spectroscopy_compiled.number_of_sequence_steps)
print(
    "Frequency axis (MHz):",
    spectroscopy_compiled.axis("frequency") / MHz,
)
"""
    ),
    code(
        """
time_ns = (
    np.arange(spectroscopy_compiled.preview(4).shape[1])
    / SAMPLE_RATE_HZ
    / ns
)

plt.figure(figsize=(12, 5))
for index in [0, 10, 20]:
    plt.plot(
        time_ns,
        spectroscopy_compiled.preview(4)[index] * 1e3,
        label=(
            f"{spectroscopy_compiled.axis('frequency')[index] / MHz:.1f} MHz"
        ),
    )
plt.xlim(0, 600)
plt.xlabel("Time (ns)")
plt.ylabel("Qubit waveform (mV)")
plt.title("Spectroscopy sequence assets")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
"""
    ),
    markdown("## Power Rabi: sequence list changes gain"),
    code(
        """
power_rabi_cfg = {
    **base_cfg,
    "gain_start": 0.0,
    "gain_stop": 0.08,
    "steps": 9,
    "qubit_len": 100 * ns,
    "qubit_sigma": 15 * ns,
}

power_rabi = PowerRabiProgram(power_rabi_cfg)
power_rabi_compiled = power_rabi.compile(
    sample_rate_hz=SAMPLE_RATE_HZ,
)

gain = power_rabi_compiled.axis("gain")
peak = np.max(np.abs(power_rabi_compiled.preview(4)), axis=1)

plt.figure(figsize=(7, 4))
plt.plot(gain, peak, "o-")
plt.xlabel("Programmed gain (V)")
plt.ylabel("Compiled waveform peak (V)")
plt.title("Power Rabi compile-time gain sweep")
plt.grid(True, alpha=0.3)
plt.show()
"""
    ),
    markdown("## T1: delay_auto sweep"),
    code(
        """
t1_cfg = {
    **base_cfg,
    "delay_start": 0,
    "delay_stop": 2 * us,
    "steps": 11,
    "pi_len": 100 * ns,
    "pi_sigma": 15 * ns,
    "pi_gain": 0.04,
}

t1 = T1Program(t1_cfg)
t1_compiled = t1.compile(sample_rate_hz=SAMPLE_RATE_HZ)

print("Delay axis (us):", t1_compiled.axis("delay") / us)
print("Fixed step duration (us):", t1_compiled.step_duration_s / us)
"""
    ),
    code(
        """
time_ns = np.arange(t1_compiled.preview(3).shape[1]) / SAMPLE_RATE_HZ / ns

plt.figure(figsize=(12, 5))
for index in [0, 5, 10]:
    plt.plot(
        time_ns,
        t1_compiled.preview(3)[index] * 1e3,
        label=f"delay={t1_compiled.axis('delay')[index] / us:.1f} us",
    )
plt.xlabel("Sequence step time (ns)")
plt.ylabel("Readout waveform (mV)")
plt.title("T1 readout moves while every sequence step remains fixed length")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
"""
    ),
    markdown("## Single-shot ground/excited sequence"),
    code(
        """
single_shot_cfg = {
    **base_cfg,
    "pi_len": 100 * ns,
    "pi_sigma": 15 * ns,
    "pi_gain": 0.04,
}

single_shot = SingleShotProgram(single_shot_cfg)
single_shot_compiled = single_shot.compile(
    sample_rate_hz=SAMPLE_RATE_HZ,
)

print("State axis:", single_shot_compiled.axis("state"))
print(
    "Ground qubit peak:",
    np.max(np.abs(single_shot_compiled.preview(4)[0])),
)
print(
    "Excited qubit peak:",
    np.max(np.abs(single_shot_compiled.preview(4)[1])),
)
"""
    ),
    markdown(
        """
## Optional hardware acquisition

Run this section only when AWG5208 and ATS9371 are connected. The compiler
uses one averaging parameter:

```python
result = compiled.acquire(n_average=1000)
```

Shot-level data remain available even after calling average helpers.
"""
    ),
    code(
        """
# from awg_alazar import AWGAlazar
#
# experiment = AWGAlazar.connect(
#     "TCPIP0::192.168.10.171::inst0::INSTR",
#     awg_sample_rate_hz=SAMPLE_RATE_HZ,
#     alazar_sample_rate_hz=1e9,
#     tone_frequency_hz=base_cfg["f_res"],
#     trigger_delay_s=0,
#     acquire_window_ns=1000,
#     integrate_window_ns=(0, 1000),
#     adc_channel="CHA",
# )
#
# compiled = power_rabi.compile(hardware=experiment)
# result = compiled.acquire(n_average=1000)
#
# print("raw:", result.raw.shape)
# print("IQ traces:", result.iq_traces.shape)
# print("single shots:", result.shots("ro").shape)
# print("trace average:", result.trace_average("ro").shape)
# print("IQ average:", result.iq_average("ro").shape)
"""
    ),
]

output = Path(__file__).with_name("demo.ipynb")
nbf.write(notebook, output)
print(output.resolve())
