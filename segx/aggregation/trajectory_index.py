"""Optional, separately toggled, *sampled* log of decoded trajectories — for
debugging and as an escape hatch if future Segregated/Dominant-
Trajectory formulas need within-trajectory co-occurrence (which the streaming
edge accumulator deliberately discards). Off by default; sampling rate is a
config value. P0: stub. P4+."""

from __future__ import annotations


class TrajectoryIndex:
    """Sampled trajectory audit log. P4+."""

    def __init__(self, sample_rate: float = 0.0) -> None:
        self.sample_rate = sample_rate
        if sample_rate:
            raise NotImplementedError("sampled trajectory index lands in P4+")
