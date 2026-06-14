"""Host-side experiment compiler for AWG5208 and ATS9371 workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from awg5200.waveforms import MIN_WAVEFORM_SAMPLES

ns = 1e-9
us = 1e-6
MHz = 1e6


class ValueExpression:
    """A compile-time value resolved from one sweep point."""

    def resolve(self, point: dict[str, Any]) -> float:
        raise NotImplementedError

    def __add__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(self, "+", as_expression(other))

    def __radd__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(as_expression(other), "+", self)

    def __sub__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(self, "-", as_expression(other))

    def __rsub__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(as_expression(other), "-", self)


ValueLike = Union[float, int, ValueExpression]


@dataclass(frozen=True)
class ConstantExpression(ValueExpression):
    value: float

    def resolve(self, point: dict[str, Any]) -> float:
        return self.value


@dataclass(frozen=True)
class SweepRef(ValueExpression):
    name: str

    def resolve(self, point: dict[str, Any]) -> float:
        try:
            return float(point[self.name])
        except KeyError as exc:
            raise KeyError(f"Unknown sweep {self.name!r}") from exc


@dataclass(frozen=True)
class BinaryExpression(ValueExpression):
    left: ValueExpression
    operator: str
    right: ValueExpression

    def resolve(self, point: dict[str, Any]) -> float:
        left = self.left.resolve(point)
        right = self.right.resolve(point)
        if self.operator == "+":
            return left + right
        if self.operator == "-":
            return left - right
        raise ValueError(f"Unsupported operator {self.operator!r}")


def as_expression(value: ValueLike) -> ValueExpression:
    if isinstance(value, ValueExpression):
        return value
    return ConstantExpression(float(value))


@dataclass(frozen=True)
class LinearSweep:
    start: float
    stop: float
    count: int

    def values(self) -> npt.NDArray[np.float64]:
        if self.count < 1:
            raise ValueError("sweep count must be positive")
        return np.linspace(self.start, self.stop, self.count)


@dataclass(frozen=True)
class ValuesSweep:
    data: tuple[Any, ...]

    def __init__(self, values: Any) -> None:
        object.__setattr__(self, "data", tuple(values))
        if not self.data:
            raise ValueError("sweep values cannot be empty")

    def values(self) -> npt.NDArray[Any]:
        return np.asarray(self.data)


@dataclass(frozen=True)
class GeneratorDeclaration:
    name: str
    channel: int
    amplitude_vpp: float


@dataclass(frozen=True)
class ReadoutDeclaration:
    name: str
    adc_channel: str | int
    length_s: float
    demod_frequency_hz: float
    marker_channel: int
    marker_number: int = 1
    marker_width_s: float = 40 * ns
    marker_low_volts: float = 0.0
    marker_high_volts: float = 1.2
    integrate_start_s: float = 0.0
    integrate_stop_s: float | None = None


@dataclass(frozen=True)
class PulseDefinition:
    name: str
    generator: str
    style: str
    length_s: ValueExpression
    frequency_hz: ValueExpression
    phase_radians: ValueExpression
    gain: ValueExpression
    sigma_s: ValueExpression | None = None
    edge_sigma_s: ValueExpression | None = None


@dataclass(frozen=True)
class PlayEvent:
    pulse_name: str
    at_s: ValueExpression | None
    when: tuple[str, Any] | None = None


@dataclass(frozen=True)
class DelayEvent:
    duration_s: ValueExpression


@dataclass(frozen=True)
class TriggerEvent:
    readout_name: str
    at_s: ValueExpression | None


ProgramEvent = Union[PlayEvent, DelayEvent, TriggerEvent]


@dataclass(frozen=True)
class ScheduledPulse:
    definition: PulseDefinition
    start_s: float
    stop_s: float
    frequency_hz: float
    phase_radians: float
    gain: float
    sigma_s: float | None
    edge_sigma_s: float | None


@dataclass(frozen=True)
class ScheduledPoint:
    coordinates: dict[str, Any]
    pulses: tuple[ScheduledPulse, ...]
    triggers_s: dict[str, float]
    duration_s: float


@dataclass
class ExperimentResult:
    """Unaveraged records plus explicit reduction helpers."""

    axes: dict[str, npt.NDArray[Any]]
    point_coordinates: tuple[dict[str, Any], ...]
    raw: npt.NDArray[np.float64]
    iq_traces: npt.NDArray[np.complex128]
    iq_shots: npt.NDArray[np.complex128]
    raw_time_s: npt.NDArray[np.float64]
    iq_time_s: npt.NDArray[np.float64]
    readout_name: str = "ro"

    def _check_readout(self, readout: str) -> None:
        if readout != self.readout_name:
            raise KeyError(f"Unknown readout {readout!r}")

    def axis(self, name: str) -> npt.NDArray[Any]:
        return self.axes[name].copy()

    def trace_average(self, readout: str = "ro") -> npt.NDArray[np.float64]:
        self._check_readout(readout)
        return np.mean(self.raw, axis=0)

    def iq_trace_average(
        self, readout: str = "ro"
    ) -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return np.mean(self.iq_traces, axis=0)

    def shots(self, readout: str = "ro") -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return self.iq_shots.copy()

    def iq_average(
        self, readout: str = "ro"
    ) -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return np.mean(self.iq_shots, axis=0)


@dataclass
class CompiledExperiment:
    """Rendered sequence assets and the record-layout contract."""

    program_name: str
    sample_rate_hz: float
    step_duration_s: float
    axes: dict[str, npt.NDArray[Any]]
    point_coordinates: tuple[dict[str, Any], ...]
    channel_waveforms: dict[int, npt.NDArray[np.float64]]
    marker_waveforms: npt.NDArray[np.bool_]
    readout: ReadoutDeclaration
    channel_amplitudes_vpp: dict[int, float]
    _hardware: Any | None = None
    _uploaded_hardware_id: int | None = None

    @property
    def number_of_sequence_steps(self) -> int:
        return len(self.point_coordinates)

    def axis(self, name: str) -> npt.NDArray[Any]:
        return self.axes[name].copy()

    def preview(self, channel: int) -> npt.NDArray[np.float64]:
        return self.channel_waveforms[channel].copy()

    def bind(self, hardware: Any) -> "CompiledExperiment":
        self._hardware = hardware
        return self

    def upload(self, hardware: Any | None = None) -> str:
        target = hardware or self._hardware
        if target is None:
            raise RuntimeError("Bind an AWGAlazar instance before upload")
        target.awg.clear_all()

        tracks: dict[int, list[str]] = {
            channel: [] for channel in self.channel_waveforms
        }
        tracks.setdefault(self.readout.marker_channel, [])

        for step_index in range(self.number_of_sequence_steps):
            for channel, waveforms in self.channel_waveforms.items():
                markers: tuple[npt.NDArray[np.bool_], ...] = ()
                if channel == self.readout.marker_channel:
                    markers = self._marker_tuple(step_index)
                asset_name = target.awg.upload_waveform_asset(
                    name=f"{self.program_name}_s{step_index:04d}_ch{channel}",
                    waveform_volts=waveforms[step_index],
                    amplitude_vpp=self.channel_amplitudes_vpp[channel],
                    markers=markers,
                )
                tracks[channel].append(asset_name)

            marker_channel = self.readout.marker_channel
            if marker_channel not in self.channel_waveforms:
                zero = np.zeros(self.marker_waveforms.shape[1])
                asset_name = target.awg.upload_waveform_asset(
                    name=f"{self.program_name}_s{step_index:04d}_marker",
                    waveform_volts=zero,
                    amplitude_vpp=0.5,
                    markers=self._marker_tuple(step_index),
                )
                tracks[marker_channel].append(asset_name)

        for channel, amplitude_vpp in self.channel_amplitudes_vpp.items():
            target.awg.set_channel_amplitude(channel, amplitude_vpp)
            target.awg.set_channel_resolution(
                channel,
                16 - self.readout.marker_number
                if channel == self.readout.marker_channel
                else 16,
            )
        if self.readout.marker_channel not in self.channel_amplitudes_vpp:
            target.awg.set_channel_amplitude(
                self.readout.marker_channel,
                0.5,
            )
            target.awg.set_channel_resolution(
                self.readout.marker_channel,
                16 - self.readout.marker_number,
            )
        target.awg.set_marker_levels(
            self.readout.marker_channel,
            self.readout.marker_number,
            self.readout.marker_low_volts,
            self.readout.marker_high_volts,
        )

        sequence_name = target.awg.create_sequence(
            self.program_name,
            tracks=tracks,
            repetitions=1,
            goto_step=1,
        )
        self._hardware = target
        self._uploaded_hardware_id = id(target)
        return sequence_name

    def _marker_tuple(
        self, step_index: int
    ) -> tuple[npt.NDArray[np.bool_], ...]:
        active = self.marker_waveforms[step_index]
        return tuple(
            active if marker == self.readout.marker_number else np.zeros_like(active)
            for marker in range(1, self.readout.marker_number + 1)
        )

    def acquire(
        self,
        n_average: int,
        *,
        hardware: Any | None = None,
        filter_type: str = "boxcar",
    ) -> ExperimentResult:
        target = hardware or self._hardware
        if target is None:
            raise RuntimeError("Bind an AWGAlazar instance before acquire")
        if self._uploaded_hardware_id != id(target):
            self.upload(target)

        raw_time_s, _, iq_time_s, _ = target.acquire_sequence_traces(
            number_of_steps=self.number_of_sequence_steps,
            number_of_averages=n_average,
            filter_type=filter_type,
        )
        raw = target.last_sequence_records_volts
        iq_traces = target.last_sequence_shot_iq
        if raw is None or iq_traces is None:
            raise RuntimeError("Sequence acquisition did not return records")

        integrate_start = max(
            0,
            int(round(self.readout.integrate_start_s * target.alazar_sample_rate_hz)),
        )
        integrate_stop_s = (
            self.readout.integrate_stop_s
            if self.readout.integrate_stop_s is not None
            else self.readout.length_s
        )
        integrate_stop = min(
            iq_traces.shape[2],
            int(round(integrate_stop_s * target.alazar_sample_rate_hz)),
        )
        if integrate_start >= integrate_stop:
            raise ValueError("Readout integration window is empty")
        iq_shots = np.mean(
            iq_traces[:, :, integrate_start:integrate_stop],
            axis=2,
        )
        return ExperimentResult(
            axes={name: values.copy() for name, values in self.axes.items()},
            point_coordinates=self.point_coordinates,
            raw=raw.copy(),
            iq_traces=iq_traces.copy(),
            iq_shots=iq_shots,
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            readout_name=self.readout.name,
        )


class ExperimentProgram:
    """Declarative program compiled into an AWG sequence and ATS record plan."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        name: str | None = None,
        final_delay_s: float = 1 * us,
    ) -> None:
        self.cfg = cfg
        self.name = name or self.__class__.__name__
        self.final_delay_s = float(final_delay_s)
        self.generators: dict[str, GeneratorDeclaration] = {}
        self.readouts: dict[str, ReadoutDeclaration] = {}
        self.sweeps: dict[str, LinearSweep | ValuesSweep] = {}
        self.pulses: dict[str, PulseDefinition] = {}
        self.events: list[ProgramEvent] = []
        self._initialize(cfg)
        self._body(cfg)
        if "ro" not in self.readouts:
            raise ValueError("Programs must declare the 'ro' readout")

    def _initialize(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def _body(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def declare_gen(
        self,
        name: str,
        *,
        ch: int,
        amplitude_vpp: float = 0.5,
    ) -> None:
        if name in self.generators:
            raise ValueError(f"Generator {name!r} is already declared")
        self.generators[name] = GeneratorDeclaration(name, ch, amplitude_vpp)

    def declare_readout(
        self,
        name: str = "ro",
        *,
        adc_channel: str | int,
        length: float,
        demod_freq: float,
        marker_channel: int = 1,
        marker_number: int = 1,
        marker_width: float = 40 * ns,
        integrate_start: float = 0.0,
        integrate_stop: float | None = None,
    ) -> None:
        if name in self.readouts:
            raise ValueError(f"Readout {name!r} is already declared")
        self.readouts[name] = ReadoutDeclaration(
            name=name,
            adc_channel=adc_channel,
            length_s=float(length),
            demod_frequency_hz=float(demod_freq),
            marker_channel=marker_channel,
            marker_number=marker_number,
            marker_width_s=float(marker_width),
            integrate_start_s=float(integrate_start),
            integrate_stop_s=(
                None if integrate_stop is None else float(integrate_stop)
            ),
        )

    def add_sweep(
        self,
        name: str,
        sweep: LinearSweep | ValuesSweep,
    ) -> SweepRef:
        if name in self.sweeps:
            raise ValueError(f"Sweep {name!r} is already declared")
        self.sweeps[name] = sweep
        return SweepRef(name)

    def add_pulse(
        self,
        name: str,
        *,
        gen: str,
        style: str,
        length: ValueLike,
        frequency: ValueLike,
        phase: ValueLike = 0.0,
        gain: ValueLike = 1.0,
        sigma: ValueLike | None = None,
        edge_sigma: ValueLike | None = None,
    ) -> None:
        if gen not in self.generators:
            raise KeyError(f"Generator {gen!r} is not declared")
        pulse_style = style.lower()
        if pulse_style not in {"const", "gaussian", "gaussian_square"}:
            raise ValueError(
                "style must be 'const', 'gaussian', or 'gaussian_square'"
            )
        self.pulses[name] = PulseDefinition(
            name=name,
            generator=gen,
            style=pulse_style,
            length_s=as_expression(length),
            frequency_hz=as_expression(frequency),
            phase_radians=as_expression(phase),
            gain=as_expression(gain),
            sigma_s=None if sigma is None else as_expression(sigma),
            edge_sigma_s=(
                None if edge_sigma is None else as_expression(edge_sigma)
            ),
        )

    def play(
        self,
        pulse_name: str,
        *,
        at: ValueLike | None = None,
        when: tuple[str, Any] | None = None,
    ) -> None:
        if pulse_name not in self.pulses:
            raise KeyError(f"Pulse {pulse_name!r} is not declared")
        if when is not None and when[0] not in self.sweeps:
            raise KeyError(f"Sweep {when[0]!r} is not declared")
        self.events.append(
            PlayEvent(
                pulse_name=pulse_name,
                at_s=None if at is None else as_expression(at),
                when=when,
            )
        )

    def delay_auto(self, duration: ValueLike) -> None:
        self.events.append(DelayEvent(as_expression(duration)))

    def trigger(
        self,
        readout: str = "ro",
        *,
        at: ValueLike | None = None,
    ) -> None:
        if readout not in self.readouts:
            raise KeyError(f"Readout {readout!r} is not declared")
        self.events.append(
            TriggerEvent(
                readout_name=readout,
                at_s=None if at is None else as_expression(at),
            )
        )

    def compile(
        self,
        *,
        sample_rate_hz: float | None = None,
        hardware: Any | None = None,
    ) -> CompiledExperiment:
        if sample_rate_hz is None:
            if hardware is None:
                raise ValueError(
                    "Provide sample_rate_hz or an AWGAlazar hardware instance"
                )
            sample_rate_hz = float(hardware.awg_sample_rate_hz)
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        points, axes = self._sweep_points()
        scheduled = tuple(self._schedule_point(point) for point in points)
        step_duration_s = max(point.duration_s for point in scheduled)
        step_samples = max(
            MIN_WAVEFORM_SAMPLES,
            int(np.ceil(step_duration_s * sample_rate_hz)),
        )

        channel_amplitudes = {
            declaration.channel: declaration.amplitude_vpp
            for declaration in self.generators.values()
        }
        channel_waveforms = {
            channel: np.zeros((len(points), step_samples), dtype=np.float64)
            for channel in channel_amplitudes
        }
        readout = self.readouts["ro"]
        markers = np.zeros((len(points), step_samples), dtype=np.bool_)

        for point_index, scheduled_point in enumerate(scheduled):
            for pulse in scheduled_point.pulses:
                generator = self.generators[pulse.definition.generator]
                values = self._render_pulse(pulse, sample_rate_hz)
                start = int(round(pulse.start_s * sample_rate_hz))
                stop = start + values.size
                channel_waveforms[generator.channel][
                    point_index, start:stop
                ] += values

            trigger_s = scheduled_point.triggers_s["ro"]
            marker_start = int(round(trigger_s * sample_rate_hz))
            marker_stop = min(
                step_samples,
                marker_start
                + max(1, int(round(readout.marker_width_s * sample_rate_hz))),
            )
            markers[point_index, marker_start:marker_stop] = True

        return CompiledExperiment(
            program_name=self.name,
            sample_rate_hz=sample_rate_hz,
            step_duration_s=step_samples / sample_rate_hz,
            axes=axes,
            point_coordinates=tuple(points),
            channel_waveforms=channel_waveforms,
            marker_waveforms=markers,
            readout=readout,
            channel_amplitudes_vpp=channel_amplitudes,
            _hardware=hardware,
        )

    def _sweep_points(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, npt.NDArray[Any]]]:
        axes = {name: sweep.values() for name, sweep in self.sweeps.items()}
        if not axes:
            return [{}], {}
        names = tuple(axes)
        points = [
            dict(zip(names, values))
            for values in product(*(axes[name] for name in names))
        ]
        return points, axes

    def _schedule_point(self, point: dict[str, Any]) -> ScheduledPoint:
        cursor_s = 0.0
        pulses: list[ScheduledPulse] = []
        triggers: dict[str, float] = {}
        for event in self.events:
            if isinstance(event, DelayEvent):
                cursor_s += event.duration_s.resolve(point)
                continue
            if isinstance(event, PlayEvent):
                if event.when is not None:
                    sweep_name, expected_value = event.when
                    if point[sweep_name] != expected_value:
                        continue
                definition = self.pulses[event.pulse_name]
                start_s = (
                    cursor_s
                    if event.at_s is None
                    else event.at_s.resolve(point)
                )
                length_s = definition.length_s.resolve(point)
                if start_s < 0 or length_s <= 0:
                    raise ValueError("Pulse start and length must be positive")
                stop_s = start_s + length_s
                pulses.append(
                    ScheduledPulse(
                        definition=definition,
                        start_s=start_s,
                        stop_s=stop_s,
                        frequency_hz=definition.frequency_hz.resolve(point),
                        phase_radians=definition.phase_radians.resolve(point),
                        gain=definition.gain.resolve(point),
                        sigma_s=(
                            None
                            if definition.sigma_s is None
                            else definition.sigma_s.resolve(point)
                        ),
                        edge_sigma_s=(
                            None
                            if definition.edge_sigma_s is None
                            else definition.edge_sigma_s.resolve(point)
                        ),
                    )
                )
                if event.at_s is None:
                    cursor_s = stop_s
                continue
            trigger_s = (
                cursor_s if event.at_s is None else event.at_s.resolve(point)
            )
            if trigger_s < 0:
                raise ValueError("Trigger time cannot be negative")
            triggers[event.readout_name] = trigger_s

        if "ro" not in triggers:
            raise ValueError("Program body must trigger the 'ro' readout")
        readout_stops = [
            trigger_s + self.readouts[name].length_s
            for name, trigger_s in triggers.items()
        ]
        duration_s = max(
            [
                cursor_s,
                *[pulse.stop_s for pulse in pulses],
                *readout_stops,
            ]
        ) + self.final_delay_s
        return ScheduledPoint(
            coordinates=point.copy(),
            pulses=tuple(pulses),
            triggers_s=triggers,
            duration_s=duration_s,
        )

    @staticmethod
    def _render_pulse(
        pulse: ScheduledPulse,
        sample_rate_hz: float,
    ) -> npt.NDArray[np.float64]:
        count = max(
            1,
            int(round((pulse.stop_s - pulse.start_s) * sample_rate_hz)),
        )
        time_s = np.arange(count, dtype=np.float64) / sample_rate_hz
        style = pulse.definition.style
        if style == "const":
            envelope = np.full(count, pulse.gain, dtype=np.float64)
        elif style == "gaussian":
            sigma_s = pulse.sigma_s
            if sigma_s is None or sigma_s <= 0:
                raise ValueError("Gaussian pulses require sigma > 0")
            center_s = time_s[-1] / 2.0
            envelope = pulse.gain * np.exp(
                -0.5 * ((time_s - center_s) / sigma_s) ** 2
            )
        else:
            sigma_s = pulse.edge_sigma_s
            if sigma_s is None or sigma_s <= 0:
                raise ValueError(
                    "Gaussian-square pulses require edge_sigma > 0"
                )
            edge_samples = max(1, int(round(3.0 * sigma_s * sample_rate_hz)))
            if 2 * edge_samples > count:
                raise ValueError(
                    "Gaussian-square pulse is too short for edge_sigma"
                )
            x = np.arange(edge_samples, dtype=np.float64)
            rise = np.exp(
                -0.5 * ((x - (edge_samples - 1)) / max(1.0, edge_samples / 3)) ** 2
            )
            envelope = np.full(count, pulse.gain, dtype=np.float64)
            envelope[:edge_samples] *= rise
            envelope[-edge_samples:] *= rise[::-1]

        if pulse.frequency_hz == 0:
            return envelope
        return envelope * np.sin(
            2.0 * np.pi * pulse.frequency_hz * time_s
            + pulse.phase_radians
        )
