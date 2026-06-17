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
# QAWG hardware demo

Workflow:

1. TOF calibration with a 1.5 us acquisition window, 0.6 us pulse, 1.0 us
   integration window, and initial trigger delay 0.
2. Resonator spectroscopy near 5.9 GHz by sweeping the SGS100A frequency.
3. Single-shot acquisition with readout gain 0.002.
4. Heterodyne tomography: estimate photon number and plot a Wigner function.
"""
    ),
    code(
        """
import numpy as np
import matplotlib.pyplot as plt

from QAWG import (
    AWGAlazar,
    ExperimentProgram,
    ExperimentResult,
    LinearSweep,
    SingleShotProgram,
    ValuesSweep,
    calculate_window,
    ns,
    us,
)
from QAWG.instrument import RohdeSchwarzSGS100A
from QAWG.tomography import (
    heterodyne_ml_density_matrix,
    normalize_heterodyne_reference,
    project_temporal_mode,
    temporal_mode_weights,
    wigner_function,
)
"""
    ),
    markdown("## Shared hardware settings"),
    code(
        """
AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
SGS100A_ADDRESS = "192.168.10.90"

AWG_SAMPLE_RATE_HZ = 2.5e9
ALAZAR_SAMPLE_RATE_HZ = 1e9
ACQUIRE_WINDOW = 1.5 * us
INTEGRATE_WINDOW = 1.0 * us

AWG_CH = 1
MARKER_CH = 1
ADC_CHANNEL = "CHB"
CHANNEL_AMPLITUDE_VPP = 0.5
IF_FREQUENCY_HZ = 50e6

TOF_N_AVERAGE = 1000
SPECTROSCOPY_N_AVERAGE = 1000
SINGLE_SHOT_N_AVERAGE = 10000
TOMOGRAPHY_N_AVERAGE = 20000
"""
    ),
    code(
        """
sgs = RohdeSchwarzSGS100A(SGS100A_ADDRESS)
sgs.frequency = 5.9e9
sgs.power = 0.0
sgs.IQ_state = "on"
sgs.pulsemod_state = "off"
sgs.configure_lo_output(True, mode="LO")
sgs.on()

experiment = AWGAlazar.connect(
    AWG_RESOURCE,
    awg_sample_rate_hz=AWG_SAMPLE_RATE_HZ,
    alazar_sample_rate_hz=ALAZAR_SAMPLE_RATE_HZ,
    acquire_window_s=ACQUIRE_WINDOW,
    trigger_slope="rising",
    trigger_level=140,
    tone_frequency_hz=IF_FREQUENCY_HZ,
    integrate_window_ns=(0.0, INTEGRATE_WINDOW / ns),
    adc_channel=ADC_CHANNEL,
    moving_average_time_s=20e-9,
    baseline_time_s=100e-9,
)

print("SGS100A:", sgs.idn())
print(f"SGS frequency: {sgs.frequency / 1e9:.9f} GHz")
print(f"SGS power: {sgs.power:.3f} dBm")
print("AWG/Alazar connected")
"""
    ),
    markdown("## 1. TOF calibration"),
    code(
        """
class TOFProgram(ExperimentProgram):
    def init(self, cfg):
        self.declare_gen(
            "readout",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["integrate_window"],
            demod_freq=cfg["if_frequency_hz"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_padding=0.0,
            integrate_time=cfg["integrate_window"],
        )
        self.add_pulse(
            "tof_pulse",
            gen="readout",
            style="gaussian_square",
            length=cfg["pulse_length"],
            edge_sigma=cfg["edge_sigma"],
            frequency=cfg["if_frequency_hz"],
            gain=cfg["readout_gain"],
            readout=True,
        )

    def body(self, cfg):
        self.play("tof_pulse", at=0)
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])


def result_from_decimate(compiled, decimated, *, initial_trigger_delay_s):
    marker_windows_s = np.zeros(
        (compiled.number_of_sequence_steps, 2),
        dtype=np.float64,
    )
    for step_index, marker in enumerate(compiled.marker_waveforms):
        active = np.flatnonzero(marker)
        if active.size:
            marker_windows_s[step_index] = (
                active[0] / compiled.sample_rate_hz,
                (active[-1] + 1) / compiled.sample_rate_hz,
            )

    iq_traces = decimated["downconverted_traces"]
    return ExperimentResult(
        axes={name: values.copy() for name, values in compiled.axes.items()},
        point_coordinates=compiled.point_coordinates,
        raw=decimated["raw_traces"],
        iq_traces=iq_traces,
        iq_shots=np.mean(iq_traces, axis=2),
        raw_time_s=decimated["raw_time_s"],
        iq_time_s=decimated["downconverted_time_s"],
        readout_name=compiled.readout.name,
        initial_trigger_delay_s=initial_trigger_delay_s,
        readout_windows_s=compiled.readout_windows_s.copy(),
        marker_windows_s=marker_windows_s,
        acquire_window_s=ACQUIRE_WINDOW,
        remove_dc_offset=getattr(compiled, "remove_dc_offset", False),
    )
"""
    ),
    code(
        """
tof_cfg = {
    "awg_ch": AWG_CH,
    "marker_ch": MARKER_CH,
    "adc_channel": ADC_CHANNEL,
    "channel_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "if_frequency_hz": IF_FREQUENCY_HZ,
    "pulse_length": 0.6 * us,
    "integrate_window": 1.0 * us,
    "trigger_delay": 0.0,
    "edge_sigma": 20 * ns,
    "readout_gain": 0.02,
}

tof_compiled = TOFProgram(tof_cfg, final_delay_s=1 * us).compile(
    hardware=experiment,
)
tof_decimated = tof_compiled.acquire_decimate(
    n_average=TOF_N_AVERAGE,
    filter_type="boxcar",
)
tof_result = result_from_decimate(
    tof_compiled,
    tof_decimated,
    initial_trigger_delay_s=tof_cfg["trigger_delay"],
)
window = calculate_window(
    tof_result,
    trigger_lead_s=20e-9,
    integration_guard_s=20e-9,
)

SUGGESTED_TRIGGER_DELAY = window.suggested_trigger_delay_s
SUGGESTED_INTEGRATE_WINDOW = (
    0.0,
    tof_cfg["integrate_window"],
)

print(f"Suggested trigger delay: {SUGGESTED_TRIGGER_DELAY / ns:.3f} ns")
print(
    "Suggested integration window:",
    f"{SUGGESTED_INTEGRATE_WINDOW[0] / ns:.3f} to",
    f"{SUGGESTED_INTEGRATE_WINDOW[1] / ns:.3f} ns",
)
"""
    ),
    markdown("## 2. Resonator spectroscopy near 5.9 GHz"),
    code(
        """
class ResonatorReadoutProgram(ExperimentProgram):
    def init(self, cfg):
        self.declare_gen(
            "readout",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["pulse_length"],
            demod_freq=cfg["if_frequency_hz"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_padding=0.0,
            integrate_time=cfg["integrate_window"],
        )
        self.add_pulse(
            "readout_pulse",
            gen="readout",
            style="const",
            length=cfg["pulse_length"],
            frequency=cfg["if_frequency_hz"],
            gain=cfg["readout_gain"],
            readout=True,
        )

    def body(self, cfg):
        self.play("readout_pulse", at=0)
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])


readout_cfg = {
    "awg_ch": AWG_CH,
    "marker_ch": MARKER_CH,
    "adc_channel": ADC_CHANNEL,
    "channel_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "if_frequency_hz": IF_FREQUENCY_HZ,
    "pulse_length": 1.0 * us,
    "integrate_window": 1.0 * us,
    "trigger_delay": SUGGESTED_TRIGGER_DELAY,
    "readout_gain": 0.02,
}

readout_compiled = ResonatorReadoutProgram(
    readout_cfg,
    final_delay_s=1 * us,
).compile(hardware=experiment)
readout_compiled.upload()
"""
    ),
    code(
        """
sgs_frequencies = np.linspace(5.88e9, 5.92e9, 81)
resonator_iq = np.empty(sgs_frequencies.size, dtype=np.complex128)

for index, frequency_hz in enumerate(sgs_frequencies):
    sgs.frequency = float(frequency_hz)
    result = readout_compiled.acquire(n_average=SPECTROSCOPY_N_AVERAGE)
    resonator_iq[index] = result["integrated_iq"][0]
    print(
        f"{index + 1:03d}/{sgs_frequencies.size}: "
        f"{frequency_hz / 1e9:.9f} GHz, "
        f"|IQ|={abs(resonator_iq[index]) * 1e3:.4f} mV"
    )

resonator_freq_ghz = sgs_frequencies / 1e9

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(resonator_freq_ghz, np.abs(resonator_iq) * 1e3, "o-")
axes[0].set_xlabel("SGS frequency (GHz)")
axes[0].set_ylabel("|IQ| (mV)")
axes[0].set_title("Resonator spectroscopy")
axes[0].grid(True, alpha=0.3)

axes[1].plot(resonator_iq.real * 1e3, resonator_iq.imag * 1e3, "o-")
axes[1].set_xlabel("I (mV)")
axes[1].set_ylabel("Q (mV)")
axes[1].set_title("IQ circle")
axes[1].axis("equal")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_index = int(np.argmin(np.abs(resonator_iq)))
RESONATOR_SGS_FREQUENCY = float(sgs_frequencies[best_index])
sgs.frequency = RESONATOR_SGS_FREQUENCY
print(f"Selected SGS frequency: {RESONATOR_SGS_FREQUENCY / 1e9:.9f} GHz")
"""
    ),
    markdown("## 3. Single shot with readout gain 0.002"),
    code(
        """
single_shot_cfg = {
    "qubit_ch": 4,
    "res_ch": AWG_CH,
    "marker_ch": MARKER_CH,
    "adc_channel": ADC_CHANNEL,
    "qubit_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "res_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "f_ge": 100e6,
    "f_res": IF_FREQUENCY_HZ,
    "pi_len": 100 * ns,
    "pi_sigma": 15 * ns,
    "pi_gain": 0.04,
    "readout_delay": 40 * ns,
    "res_len": 1.0 * us,
    "ro_len": 1.0 * us,
    "res_gain": 0.002,
    "integrate_time": 1.0 * us,
    "trigger_delay": SUGGESTED_TRIGGER_DELAY,
}

single_shot_compiled = SingleShotProgram(
    single_shot_cfg,
    final_delay_s=1 * us,
).compile(hardware=experiment)
single_shot_result = single_shot_compiled.acquire(
    n_average=SINGLE_SHOT_N_AVERAGE,
)

states = single_shot_result["axes"]["state"]
shots = single_shot_result["shot_iq"]

print("States:", states)
print("Shot IQ shape:", shots.shape)
print("Readout gain:", single_shot_cfg["res_gain"])

plt.figure(figsize=(6, 6))
for step, state in enumerate(states):
    plt.scatter(
        shots[:, step].real * 1e3,
        shots[:, step].imag * 1e3,
        s=6,
        alpha=0.35,
        label=str(state),
    )
plt.xlabel("I (mV)")
plt.ylabel("Q (mV)")
plt.title("Single-shot IQ, readout gain 0.002")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend(title="State")
plt.show()
"""
    ),
    markdown("## 4. Photon number and Wigner function"),
    code(
        """
class TomographyProgram(ExperimentProgram):
    def init(self, cfg):
        self.declare_gen(
            "readout",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        state = self.add_sweep("state", ValuesSweep(("reference", "signal")))
        self.state = state
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["readout_length"],
            demod_freq=cfg["if_frequency_hz"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_padding=0.0,
            integrate_time=cfg["readout_length"],
        )
        self.add_pulse(
            "signal_pulse",
            gen="readout",
            style="const",
            length=cfg["readout_length"],
            frequency=cfg["if_frequency_hz"],
            gain=cfg["signal_gain"],
            readout=True,
        )
        self.add_pulse(
            "reference_pulse",
            gen="readout",
            style="const",
            length=cfg["readout_length"],
            frequency=cfg["if_frequency_hz"],
            gain=0.0,
            readout=True,
        )

    def body(self, cfg):
        self.play("reference_pulse", at=0, when=("state", "reference"))
        self.play("signal_pulse", at=0, when=("state", "signal"))
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])


tomography_cfg = {
    "awg_ch": AWG_CH,
    "marker_ch": MARKER_CH,
    "adc_channel": ADC_CHANNEL,
    "channel_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "if_frequency_hz": IF_FREQUENCY_HZ,
    "readout_length": 1.0 * us,
    "trigger_delay": SUGGESTED_TRIGGER_DELAY,
    "signal_gain": 0.002,
}

tomography_compiled = TomographyProgram(
    tomography_cfg,
    final_delay_s=1 * us,
).compile(hardware=experiment)
tomography_debug = tomography_compiled.acquire_decimate(
    n_average=TOMOGRAPHY_N_AVERAGE,
    filter_type="boxcar",
)

iq_traces = tomography_debug["downconverted_traces"]
iq_time_s = tomography_debug["downconverted_time_s"]
mode_samples = int(round(tomography_cfg["readout_length"] * ALAZAR_SAMPLE_RATE_HZ))
mode_samples = min(mode_samples, iq_traces.shape[2])
weights = temporal_mode_weights(mode_samples, kind="boxcar")

reference_samples = project_temporal_mode(iq_traces[:, 0, :], weights)
signal_samples = project_temporal_mode(iq_traces[:, 1, :], weights)
reference_alpha, (signal_alpha,), iq_offset, iq_scale = normalize_heterodyne_reference(
    reference_samples,
    signal_samples,
)

rho = heterodyne_ml_density_matrix(
    signal_alpha,
    cutoff=8,
    iterations=200,
    dilution=0.5,
)
number = np.arange(rho.shape[0], dtype=float)
photon_number = float(np.real(np.sum(number * np.diag(rho))))
purity = float(np.real(np.trace(rho @ rho)))

print("Projected reference samples:", reference_alpha.shape)
print("Projected signal samples:", signal_alpha.shape)
print(f"Photon number <n>: {photon_number:.4f}")
print(f"Purity Tr(rho^2): {purity:.4f}")
print(f"IQ offset: {iq_offset}")
print(f"IQ scale: {iq_scale}")
"""
    ),
    code(
        """
x = np.linspace(-3.0, 3.0, 81)
y = np.linspace(-3.0, 3.0, 81)
W = wigner_function(rho, x, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(
    reference_alpha.real,
    reference_alpha.imag,
    s=3,
    alpha=0.15,
    label="reference",
)
axes[0].scatter(
    signal_alpha.real,
    signal_alpha.imag,
    s=3,
    alpha=0.15,
    label="signal",
)
axes[0].set_xlabel("Re(alpha)")
axes[0].set_ylabel("Im(alpha)")
axes[0].set_title("Normalized heterodyne samples")
axes[0].axis("equal")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

image = axes[1].imshow(
    W,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin="lower",
    cmap="RdBu_r",
    aspect="equal",
)
axes[1].set_xlabel("Re(alpha)")
axes[1].set_ylabel("Im(alpha)")
axes[1].set_title(f"Wigner function, <n>={photon_number:.3f}")
fig.colorbar(image, ax=axes[1], label="W(alpha)")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## Close hardware sessions"),
    code(
        """
if "sgs" in globals():
    sgs.off()
    sgs.configure_lo_output(False)
    sgs.close()
    print("SGS100A RF and rear LO outputs disabled")

if "experiment" in globals():
    experiment.close()
    print("AWG VISA session closed")
"""
    ),
]

output = Path(__file__).resolve().parent.parent / "demo.ipynb"
nbf.write(notebook, output)
print(output.resolve())
