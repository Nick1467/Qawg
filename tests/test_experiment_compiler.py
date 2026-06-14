from __future__ import annotations

import unittest

import numpy as np

from QAWG import (
    ExperimentProgram,
    ExperimentResult,
    LinearSweep,
    SingleShotProgram,
    ns,
    us,
)
from QAWG import PowerRabiProgram, T1Program


class DelayProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("drive", ch=3)
        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=1 * us,
            demod_freq=50e6,
        )
        self.delay = self.add_sweep(
            "delay",
            LinearSweep(0, 200 * ns, 6),
        )
        self.add_pulse(
            "pulse",
            gen="drive",
            style="gaussian",
            length=100 * ns,
            sigma=15 * ns,
            frequency=50e6,
            gain=0.02,
        )

    def _body(self, cfg):
        self.delay_auto(self.delay)
        self.play("pulse")
        self.trigger("ro", at=0)


class ExperimentCompilerTests(unittest.TestCase):
    def test_delay_sweep_renders_trace_by_trace_steps(self) -> None:
        compiled = DelayProgram({}).compile(sample_rate_hz=2.5e9)

        self.assertEqual(compiled.number_of_sequence_steps, 6)
        np.testing.assert_allclose(
            compiled.axis("delay") / ns,
            [0, 40, 80, 120, 160, 200],
        )
        self.assertEqual(compiled.preview(3).shape[0], 6)

        starts = []
        for trace in compiled.preview(3):
            starts.append(np.flatnonzero(np.abs(trace) > 1e-5)[0])
        np.testing.assert_array_equal(np.diff(starts), 100)

    def test_power_rabi_changes_waveform_gain_per_step(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "gain_start": 0.01,
            "gain_stop": 0.05,
            "steps": 5,
            "qubit_len": 100 * ns,
            "qubit_sigma": 15 * ns,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = PowerRabiProgram(cfg).compile(sample_rate_hz=2.5e9)
        peaks = np.max(np.abs(compiled.preview(4)), axis=1)

        self.assertEqual(compiled.number_of_sequence_steps, 5)
        self.assertTrue(np.all(np.diff(peaks) > 0))

    def test_t1_uses_fixed_step_length_for_all_delays(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "delay_start": 0,
            "delay_stop": 2 * us,
            "steps": 5,
            "pi_len": 100 * ns,
            "pi_sigma": 15 * ns,
            "pi_gain": 0.02,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = T1Program(cfg).compile(sample_rate_hz=2.5e9)

        shapes = {values.shape[1] for values in compiled.channel_waveforms.values()}
        self.assertEqual(shapes, {compiled.preview(3).shape[1]})
        self.assertEqual(compiled.number_of_sequence_steps, 5)

    def test_result_keeps_shots_and_averages_explicitly(self) -> None:
        raw = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
        iq_traces = raw.astype(complex)
        iq_shots = np.mean(iq_traces, axis=2)
        result = ExperimentResult(
            axes={"gain": np.array([0.1, 0.2])},
            point_coordinates=({"gain": 0.1}, {"gain": 0.2}),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=iq_shots,
            raw_time_s=np.arange(4),
            iq_time_s=np.arange(4),
        )

        self.assertEqual(result.shots("ro").shape, (3, 2))
        np.testing.assert_allclose(result.trace_average("ro"), raw.mean(axis=0))
        np.testing.assert_allclose(result.iq_average("ro"), iq_shots.mean(axis=0))

    def test_single_shot_ground_and_excited_steps_differ(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "pi_len": 100 * ns,
            "pi_sigma": 15 * ns,
            "pi_gain": 0.02,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = SingleShotProgram(cfg).compile(sample_rate_hz=2.5e9)

        self.assertEqual(compiled.axis("state").tolist(), ["g", "e"])
        ground = compiled.preview(4)[0]
        excited = compiled.preview(4)[1]
        self.assertTrue(np.all(ground == 0))
        self.assertGreater(np.max(np.abs(excited)), 0)

    def test_acquire_uses_one_n_average_axis(self) -> None:
        compiled = DelayProgram({}).compile(sample_rate_hz=2.5e9)

        class FakeHardware:
            alazar_sample_rate_hz = 1e9

            def acquire_sequence_traces(
                self,
                number_of_steps,
                number_of_averages,
                filter_type,
            ):
                self.called_with = (
                    number_of_steps,
                    number_of_averages,
                    filter_type,
                )
                self.last_sequence_records_volts = np.ones(
                    (number_of_averages, number_of_steps, 128)
                )
                self.last_sequence_shot_iq = np.ones(
                    (number_of_averages, number_of_steps, 100),
                    dtype=complex,
                )
                return (
                    np.arange(128) / 1e9,
                    np.ones((number_of_steps, 128)),
                    np.arange(100) / 1e9,
                    np.ones((number_of_steps, 100), dtype=complex),
                )

        hardware = FakeHardware()
        compiled._hardware = hardware
        compiled._uploaded_hardware_id = id(hardware)

        result = compiled.acquire(n_average=7)

        self.assertEqual(hardware.called_with, (6, 7, "boxcar"))
        self.assertEqual(result.raw.shape, (7, 6, 128))
        self.assertEqual(result.shots("ro").shape, (7, 6))
        self.assertEqual(result.trace_average("ro").shape, (6, 128))


if __name__ == "__main__":
    unittest.main()
