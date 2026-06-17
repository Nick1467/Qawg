from __future__ import annotations

from pathlib import Path

import nbformat


path = Path("resonator_tof.ipynb")
nb = nbformat.read(path, as_version=4)


def find_markdown(title: str) -> int:
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and cell.source.strip() == title:
            return index
    raise RuntimeError(f"Could not find markdown cell {title!r}")


step3 = find_markdown("## Step 3: compile, upload once, and acquire raw shots")
step4 = find_markdown("## Step 4: manually demodulate each shot")
step5 = find_markdown("## Step 5: inspect records and temporal-mode window")
step6 = find_markdown("## Step 6: project the temporal mode and normalize")
step7 = find_markdown("## Step 7: reconstruct density matrix and Wigner function")

new_step3_to_6 = [
    nbformat.v4.new_markdown_cell("## Step 3: compile and upload once"),
    nbformat.v4.new_code_cell(
        """tomo_program = ResonatorTomographySequenceProgram(
    tomo_cfg,
    name="resonator_tomography_sequence",
    final_delay_s=FINAL_DELAY,
)
tomo_program.REMOVE_DC_OFFSET = True
tomo_compiled = tomo_program.compile(hardware=experiment)

print("Sequence steps:", tomo_compiled.number_of_sequence_steps)
print("Gain axis:", tomo_compiled.axis("readout_gain"))

# Upload the two-step sequence once. Acquisition below is chunked and projects
# each shot immediately, so it does not keep a huge (shot, step, time) IQ array.
tomo_compiled.upload(hardware=experiment)
"""
    ),
    nbformat.v4.new_markdown_cell(
        "## Step 4: chunked acquisition and temporal-mode projection"
    ),
    nbformat.v4.new_code_cell(
        """def acquire_sequence_mode_samples(
    compiled,
    *,
    n_shots,
    mode_start_ns,
    mode_stop_ns,
    demod_frequency_hz,
    batch_shots=100,
    remove_dc_offset=True,
):
    steps = compiled.number_of_sequence_steps
    mode_start = experiment.ns2cycles(mode_start_ns, inst="adc")
    mode_stop = experiment.ns2cycles(mode_stop_ns, inst="adc")
    if mode_stop <= mode_start:
        raise ValueError("mode_stop_ns must be greater than mode_start_ns")

    samples_per_record = experiment.acquire_window_cycles
    mode_stop = min(mode_stop, samples_per_record)
    mode_samples = mode_stop - mode_start
    if mode_samples <= 0:
        raise ValueError("temporal mode is outside the acquired record")

    mode = temporal_mode_weights(mode_samples, kind="boxcar")
    time_s = np.arange(samples_per_record) / experiment.alazar_sample_rate_hz
    mode_time_s = time_s[mode_start:mode_stop]
    demod = np.exp(-1j * 2 * np.pi * demod_frequency_hz * mode_time_s)
    weights = np.conjugate(mode) * demod

    mode_values = np.empty((n_shots, steps), dtype=np.complex128)
    average_records = np.zeros((steps, samples_per_record), dtype=np.float64)

    written = 0
    while written < n_shots:
        current = min(batch_shots, n_shots - written)
        records = experiment._capture_records(n_average=current * steps)
        records = records.reshape(current, steps, records.shape[1])
        if remove_dc_offset:
            records = records - np.mean(records, axis=2, keepdims=True)

        average_records += np.sum(records, axis=0)
        mode_slice = records[:, :, mode_start:mode_stop]
        mode_values[written:written + current] = 2.0 * np.einsum(
            "bsm,m->bs",
            mode_slice,
            weights,
            optimize=True,
        )
        written += current
        print(f"Acquired {written}/{n_shots} shots")

    average_records /= n_shots
    average_iq_traces = (
        2.0
        * average_records
        * np.exp(-1j * 2 * np.pi * demod_frequency_hz * time_s)[None, :]
    )
    return mode_values, average_iq_traces, time_s, (mode_start, mode_stop), mode


TOMO_BATCH_SHOTS = 100

tomo_mode_values, tomo_average_iq_traces, tomo_raw_time_s, tomo_mode_indices, mode = (
    acquire_sequence_mode_samples(
        tomo_compiled,
        n_shots=TOMO_N_SHOTS,
        mode_start_ns=TOMO_MODE_START_NS,
        mode_stop_ns=TOMO_MODE_STOP_NS,
        demod_frequency_hz=tomo_cfg["frequency"],
        batch_shots=TOMO_BATCH_SHOTS,
        remove_dc_offset=tomo_compiled.remove_dc_offset,
    )
)

reference_mode_volts = tomo_mode_values[:, 0]
signal_mode_volts = tomo_mode_values[:, 1]
reference_average = tomo_average_iq_traces[0]
signal_average = tomo_average_iq_traces[1]
tomo_iq_time_ns = tomo_raw_time_s / ns

print("Mode samples (shot, step):", tomo_mode_values.shape)
print("Average IQ traces (step, time):", tomo_average_iq_traces.shape)
print("Mode index window:", tomo_mode_indices)
print(
    "Reference mode mean: "
    f"{np.mean(reference_mode_volts).real * 1e3:.6f} + "
    f"{np.mean(reference_mode_volts).imag * 1e3:.6f}j mV"
)
print(
    "Signal mode mean: "
    f"{np.mean(signal_mode_volts).real * 1e3:.6f} + "
    f"{np.mean(signal_mode_volts).imag * 1e3:.6f}j mV"
)
"""
    ),
    nbformat.v4.new_markdown_cell("## Step 5: inspect averaged records and temporal-mode window"),
    nbformat.v4.new_code_cell(
        """fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(tomo_iq_time_ns, reference_average.real * 1e3, label="ref I")
axes[0].plot(tomo_iq_time_ns, reference_average.imag * 1e3, label="ref Q")
axes[0].plot(tomo_iq_time_ns, np.abs(reference_average) * 1e3, label="ref |IQ|")
axes[0].axvspan(TOMO_MODE_START_NS, TOMO_MODE_STOP_NS, color="tab:green", alpha=0.15)
axes[0].set_title("Reference average")
axes[0].set_ylabel("IQ voltage (mV)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(tomo_iq_time_ns, signal_average.real * 1e3, label="signal I")
axes[1].plot(tomo_iq_time_ns, signal_average.imag * 1e3, label="signal Q")
axes[1].plot(tomo_iq_time_ns, np.abs(signal_average) * 1e3, label="signal |IQ|")
axes[1].axvspan(TOMO_MODE_START_NS, TOMO_MODE_STOP_NS, color="tab:green", alpha=0.15)
axes[1].set_title("Signal average and temporal mode window")
axes[1].set_xlabel("Time after ATS trigger (ns)")
axes[1].set_ylabel("IQ voltage (mV)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
"""
    ),
    nbformat.v4.new_markdown_cell("## Step 6: normalize projected heterodyne samples"),
    nbformat.v4.new_code_cell(
        """alpha_reference, (alpha_signal,), iq_offset, iq_scale = (
    normalize_heterodyne_reference(reference_mode_volts, signal_mode_volts)
)

print("Mode samples:", mode.size)
print("IQ offset (mode volts):", iq_offset)
print("IQ scale (mode volts / alpha):", iq_scale)
print("Reference <|alpha|^2>:", np.mean(np.abs(alpha_reference) ** 2))
print("Signal mean alpha:", np.mean(alpha_signal))
print("Signal <n> estimate from samples:", np.mean(np.abs(alpha_signal) ** 2) - 1)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(alpha_reference.real, alpha_reference.imag, s=5, alpha=0.15, label="reference")
ax.scatter(alpha_signal.real, alpha_signal.imag, s=5, alpha=0.15, label="signal")
ax.set_xlabel("Re(alpha)")
ax.set_ylabel("Im(alpha)")
ax.set_title("Temporal-mode heterodyne samples")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
"""
    ),
]

nb.cells[step3:step7] = new_step3_to_6

nbformat.write(nb, path)
print(path.resolve())
