"""Averaging and record-layout helpers for acquired QAWG data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .compiler import CompiledExperiment, ExperimentResult


@dataclass(frozen=True)
class CompiledAverages:
    raw: npt.NDArray[np.float64]
    iq_traces: npt.NDArray[np.complex128]
    iq_shots: npt.NDArray[np.complex128]


def validate_average_count(n_average: int) -> int:
    averages = int(n_average)
    if averages < 1:
        raise ValueError("n_average must be positive")
    return averages


def remove_record_dc_offset(
    records: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return records - np.mean(records, axis=1, keepdims=True)


def integrate_shots(
    iq_traces: npt.NDArray[np.complex128],
    integration_window: slice,
) -> npt.NDArray[np.complex128]:
    return np.mean(iq_traces[:, integration_window], axis=1)


def average_shots(
    shots: npt.NDArray[np.complex128],
) -> np.complex128:
    return np.complex128(np.mean(shots))


def compiled_averages(
    *,
    records: npt.NDArray[np.float64],
    downconverted: npt.NDArray[np.complex128],
    number_of_steps: int,
    n_average: int,
    integration_window: slice,
) -> CompiledAverages:
    averages = validate_average_count(n_average)
    steps = int(number_of_steps)
    if steps < 1:
        raise ValueError("number_of_steps must be positive")

    expected_records = averages * steps
    if records.shape[0] != expected_records:
        raise ValueError("records do not match n_average * number_of_steps")
    if downconverted.shape[0] != expected_records:
        raise ValueError("downconverted traces do not match acquired records")

    raw = records.reshape(averages, steps, records.shape[1])
    iq_traces = downconverted.reshape(
        averages,
        steps,
        downconverted.shape[1],
    )
    iq_shots = np.mean(iq_traces[:, :, integration_window], axis=2)
    return CompiledAverages(raw=raw, iq_traces=iq_traces, iq_shots=iq_shots)


def trace_average(
    result: "ExperimentResult",
    readout: str = "ro",
) -> npt.NDArray[np.float64]:
    result._check_readout(readout)
    return np.mean(result.raw, axis=0)


def iq_trace_average(
    result: "ExperimentResult",
    readout: str = "ro",
) -> npt.NDArray[np.complex128]:
    result._check_readout(readout)
    return np.mean(result.iq_traces, axis=0)


def shots(
    result: "ExperimentResult",
    readout: str = "ro",
) -> npt.NDArray[np.complex128]:
    result._check_readout(readout)
    return result.iq_shots.copy()


def iq_average(
    result: "ExperimentResult",
    readout: str = "ro",
) -> npt.NDArray[np.complex128]:
    result._check_readout(readout)
    return np.mean(result.iq_shots, axis=0)
