from __future__ import annotations

import unittest

import numpy as np

from QAWG.tomography import (
    calibrate_iq_samples,
    coherent_density_matrix,
    heterodyne_ml_density_matrix,
    normalize_heterodyne_reference,
    project_temporal_mode,
    temporal_mode_weights,
    wigner_function,
)


class TomographyTests(unittest.TestCase):
    def test_temporal_mode_projection(self) -> None:
        weights = temporal_mode_weights(4, kind="boxcar")
        traces = np.zeros((2, 8), dtype=np.complex128)
        traces[:, 2:6] = np.array([[1.0], [2.0j]])
        projected = project_temporal_mode(traces, weights, start_sample=2)
        np.testing.assert_allclose(projected, [2.0, 4.0j])

    def test_exponential_mode_uses_field_amplitude_decay(self) -> None:
        weights = temporal_mode_weights(
            11,
            kind="exponential",
            decay_samples=5,
        )
        self.assertAlmostEqual(abs(weights[5] / weights[0]), np.exp(-0.5))
        self.assertAlmostEqual(float(np.sum(np.abs(weights) ** 2)), 1.0)

    def test_iq_calibration_maps_reference_and_signal_means(self) -> None:
        reference = np.array([1 + 2j, 1 + 2j])
        signal = np.array([3 + 4j, 3 + 4j])
        calibrated_reference, calibrated_signal, _, _ = calibrate_iq_samples(
            reference,
            signal,
            target_alpha=2j,
        )
        self.assertAlmostEqual(abs(np.mean(calibrated_reference)), 0.0)
        self.assertAlmostEqual(abs(np.mean(calibrated_signal) - 2j), 0.0)

    def test_reference_normalization_sets_unit_complex_variance(self) -> None:
        reference = np.array([-1.0, 1.0, -1.0j, 1.0j]) + (2.0 + 3.0j)
        normalized, (signal,), offset, scale = normalize_heterodyne_reference(
            reference,
            reference + 2.0,
        )
        self.assertAlmostEqual(abs(np.mean(normalized)), 0.0)
        self.assertAlmostEqual(float(np.mean(np.abs(normalized) ** 2)), 1.0)
        self.assertAlmostEqual(offset, 2.0 + 3.0j)
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(np.mean(signal), 2.0)

    def test_ml_density_matrix_is_physical(self) -> None:
        rng = np.random.default_rng(1234)
        alpha = 0.6 + 0.2j
        samples = alpha + (
            rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
        ) / np.sqrt(2.0)
        rho = heterodyne_ml_density_matrix(
            samples,
            cutoff=6,
            iterations=80,
        )
        self.assertAlmostEqual(float(np.real(np.trace(rho))), 1.0, places=10)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(rho))), -1e-10)
        expected = coherent_density_matrix(alpha, cutoff=6)
        fidelity = float(np.real(np.trace(rho @ expected)))
        self.assertGreater(fidelity, 0.75)

    def test_vacuum_wigner_origin(self) -> None:
        rho = coherent_density_matrix(0.0, cutoff=5)
        wigner = wigner_function(rho, [0.0], [0.0])
        self.assertAlmostEqual(wigner[0, 0], 2.0 / np.pi, places=12)


if __name__ == "__main__":
    unittest.main()
