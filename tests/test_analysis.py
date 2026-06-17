import unittest

import numpy as np

from QAWG import ExperimentResult, calculate_window, diagnose_phase_shots


class WindowAnalysisTests(unittest.TestCase):
    def test_calculate_window_uses_readout_duration_not_late_transient(self):
        raw_time_s = np.arange(1500) / 1e9
        iq_time_s = np.arange(1481) / 1e9
        envelope = np.zeros(iq_time_s.size)
        envelope[120:720] = 1.0
        envelope[1100:1110] = 2.0
        raw = np.zeros((4, 1, raw_time_s.size))
        iq_traces = np.tile(
            envelope.astype(complex),
            (4, 1, 1),
        )
        result = ExperimentResult(
            axes={},
            point_coordinates=({},),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=np.mean(iq_traces, axis=2),
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            initial_trigger_delay_s=500e-9,
            readout_windows_s=np.array([[500e-9, 1100e-9]]),
            marker_windows_s=np.array([[0.0, 1600e-9]]),
            acquire_window_s=1500e-9,
            remove_dc_offset=True,
        )

        analysis = calculate_window(
            result,
            plot=False,
            report=False,
        )

        self.assertAlmostEqual(
            analysis.suggested_trigger_delay_s,
            600e-9,
        )
        self.assertAlmostEqual(
            analysis.integration_stop_s,
            640e-9,
        )
        self.assertTrue(result.remove_dc_offset)

    def test_calculate_window_smooths_downconversion_ripple(self):
        raw_time_s = np.arange(1500) / 1e9
        iq_time_s = np.arange(1500) / 1e9
        carrier = np.exp(-1j * 2 * np.pi * 100e6 * iq_time_s)
        baseband = np.zeros(iq_time_s.size, dtype=np.complex128)
        baseband[120:720] = 1.0 + carrier[120:720]
        raw = np.zeros((4, 1, raw_time_s.size))
        iq_traces = np.tile(baseband, (4, 1, 1))
        result = ExperimentResult(
            axes={},
            point_coordinates=({},),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=np.mean(iq_traces[:, :, :600], axis=2),
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            initial_trigger_delay_s=500e-9,
            readout_windows_s=np.array([[500e-9, 1100e-9]]),
            marker_windows_s=np.array([[0.0, 1600e-9]]),
            acquire_window_s=1500e-9,
        )

        analysis = calculate_window(
            result,
            plot=False,
            report=False,
            envelope_smoothing_s=20e-9,
        )

        self.assertAlmostEqual(
            analysis.suggested_trigger_delay_s,
            600e-9,
            delta=2e-9,
        )

    def test_calculate_window_ignores_short_marker_feedthrough(self):
        raw_time_s = np.arange(1500) / 1e9
        iq_time_s = np.arange(1500) / 1e9
        envelope = np.zeros(iq_time_s.size)
        envelope[0:8] = 0.15
        envelope[400:920] = 0.22
        envelope[1210:1220] = 0.12
        raw = np.zeros((4, 1, raw_time_s.size))
        iq_traces = np.tile(envelope.astype(complex), (4, 1, 1))
        result = ExperimentResult(
            axes={},
            point_coordinates=({},),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=np.mean(iq_traces, axis=2),
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            initial_trigger_delay_s=0.0,
            readout_windows_s=np.array([[0.0, 1000e-9]]),
            marker_windows_s=np.array([[0.0, 1200e-9]]),
            acquire_window_s=1500e-9,
        )

        analysis = calculate_window(
            result,
            plot=False,
            report=False,
        )

        self.assertAlmostEqual(
            analysis.measured_rise_s,
            400e-9,
            delta=2e-9,
        )
        self.assertAlmostEqual(
            analysis.suggested_trigger_delay_s,
            380e-9,
            delta=2e-9,
        )

    def test_diagnose_phase_shots_reports_opposite_steps(self):
        phases = np.array([0.0, np.pi])
        shot_phase = np.deg2rad(20.0)
        point = np.exp(1j * shot_phase)
        iq_traces = np.ones((8, 2, 10), dtype=np.complex128)
        iq_traces[:, 0, :] *= point
        iq_traces[:, 1, :] *= -point
        result = ExperimentResult(
            axes={"phase": phases},
            point_coordinates=({"phase": 0.0}, {"phase": np.pi}),
            raw=np.zeros((8, 2, 10)),
            iq_traces=iq_traces,
            iq_shots=np.mean(iq_traces, axis=2),
            raw_time_s=np.arange(10) / 1e9,
            iq_time_s=np.arange(10) / 1e9,
        )

        diagnostics = diagnose_phase_shots(result, report=False)

        self.assertLess(diagnostics.opposite_error_percent, 1e-12)
        self.assertLess(diagnostics.common_phase_jitter_degrees, 1e-12)


if __name__ == "__main__":
    unittest.main()
