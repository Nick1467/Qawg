"""Compatibility exports for timeline helpers.

The implementation lives in :mod:`QAWG.timeline` because timeline rendering is
part of the experiment description layer, not the AWG5200 hardware boundary.
"""

from ..timeline import (
    Delay,
    Parallel,
    Timeline,
    Waveform,
    align_channel_envelopes,
    align_channels,
    channel_names,
    delay,
    delay_auto,
    parallel,
    waveform,
)

__all__ = [
    "Delay",
    "Parallel",
    "Timeline",
    "Waveform",
    "align_channel_envelopes",
    "align_channels",
    "channel_names",
    "delay",
    "delay_auto",
    "parallel",
    "waveform",
]
