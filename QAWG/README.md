# QAWG Experiment Compiler

`QAWG` is a host-side experiment compiler for the Tektronix AWG5208 and
ATS9371 stack in this repository.

It intentionally resembles QICK's program style:

```python
class MyProgram(ExperimentProgram):
    def _initialize(self, cfg):
        ...

    def _body(self, cfg):
        ...
```

The execution model is different from FPGA firmware. Sweeps are expanded at
compile time into AWG waveform assets and sequence-list steps. ATS records are
captured in the same interleaved order and reshaped using the compiler's record
layout.

## Data contract

The only acquisition averaging parameter is `n_average`:

```python
result = compiled.acquire(n_average=1000)
```

For `P` sequence points, the result keeps the unaveraged records:

```python
result.raw.shape
# (n_average, P, adc_sample)

result.iq_traces.shape
# (n_average, P, iq_sample)

result.shots("ro").shape
# (n_average, P)
```

Averaging is explicit:

```python
raw_average = result.trace_average("ro")
iq_trace_average = result.iq_trace_average("ro")
iq_average = result.iq_average("ro")
```

This is important for single-shot experiments because the compiler does not
discard shot-level information.

## Basic program

```python
from QAWG import ExperimentProgram, LinearSweep, MHz, ns, us


class SpectroscopyProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("qubit", ch=4, amplitude_vpp=0.5)
        self.declare_gen("res", ch=3, amplitude_vpp=0.5)

        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=1 * us,
            demod_freq=cfg["f_res"],
            marker_channel=1,
        )

        frequency = self.add_sweep(
            "frequency",
            LinearSweep(2920 * MHz, 3000 * MHz, 101),
        )

        self.add_pulse(
            "probe",
            gen="qubit",
            style="const",
            length=15 * us,
            frequency=frequency,
            gain=0.01,
        )
        self.add_pulse(
            "readout",
            gen="res",
            style="const",
            length=2 * us,
            frequency=cfg["f_res"],
            gain=0.02,
        )

    def _body(self, cfg):
        self.play("probe", at=0)
        self.play("readout", at=0)
        self.trigger("ro", at=650 * ns)
```

Compile without hardware to inspect generated assets:

```python
program = SpectroscopyProgram(
    {"f_res": 50 * MHz},
    final_delay_s=1 * us,
)

compiled = program.compile(sample_rate_hz=2.5e9)
frequencies = compiled.axis("frequency")
qubit_steps = compiled.preview(channel=4)
```

Compile and bind to an existing `AWGAlazar` instance:

```python
compiled = program.compile(hardware=experiment)
result = compiled.acquire(n_average=1000)
```

## Relative timing

`play()` without `at` advances a program cursor:

```python
self.play("pi")
self.delay_auto(40 * ns)
self.play("readout")
self.trigger("ro")
```

This compiles to:

```text
pi pulse -> 40 ns delay -> readout + marker
```

Explicit `at` allows overlapping pulses:

```python
self.play("probe", at=0)
self.play("readout", at=0)
self.trigger("ro", at=650 * ns)
```

## Power Rabi

Power Rabi sweeps waveform gain. AWG5208 cannot update an FPGA gain register
per shot, so the compiler renders one waveform asset per gain and places those
assets in the sequence list.

```python
from QAWG import PowerRabiProgram

program = PowerRabiProgram({
    "f_res": 50e6,
    "f_ge": 100e6,
    "gain_start": 0.0,
    "gain_stop": 0.1,
    "steps": 51,
    "qubit_len": 100 * ns,
    "qubit_sigma": 15 * ns,
    "res_len": 1 * us,
    "res_gain": 0.02,
    "ro_len": 1 * us,
})

compiled = program.compile(hardware=experiment)
result = compiled.acquire(n_average=1000)

gain = result.axis("gain")
iq = result.iq_average("ro")
```

## T1

T1 sweeps `delay_auto()` between the pi pulse and readout. Every compiled
sequence step is padded to the same total duration, so trigger cadence does not
depend on the delay value.

```python
from QAWG import T1Program

program = T1Program({
    "f_res": 50e6,
    "f_ge": 100e6,
    "delay_start": 0,
    "delay_stop": 100 * us,
    "steps": 101,
    "pi_len": 100 * ns,
    "pi_sigma": 15 * ns,
    "pi_gain": 0.02,
    "res_len": 1 * us,
    "res_gain": 0.02,
    "ro_len": 1 * us,
})

result = program.compile(hardware=experiment).acquire(n_average=1000)
delay = result.axis("delay")
iq = result.iq_average("ro")
```

## Single shot

Single-shot calibration uses a categorical `state` axis. The `g` step skips
the pi pulse; the `e` step plays it.

```python
from QAWG import SingleShotProgram

compiled = SingleShotProgram(cfg).compile(hardware=experiment)
result = compiled.acquire(n_average=10_000)

states = result.axis("state")
shots = result.shots("ro")

ground = shots[:, 0]
excited = shots[:, 1]
```

No automatic average is performed by `shots()`.

## Current limits

- Sweeps are compile-time AWG sequence expansion, not FPGA runtime loops.
- The current backend supports one readout named `"ro"`.
- Sweep axes use Cartesian expansion when more than one axis is declared.
- Conditional playback currently supports equality checks such as
  `when=("state", "e")`.
- Waveform upload caching and sequence chunking are future optimizations.
- Hardware acquisition requires an already configured `AWGAlazar` instance
  whose sample rates and readout settings match the compiled program.
