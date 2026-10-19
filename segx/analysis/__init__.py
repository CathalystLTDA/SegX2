"""Analyses of the **input tables**, before anything is routed.

:mod:`segx.metrics` measures what people experience along trajectories and needs
a street network to exist at all. This package is the other half: what the
supplied data already says on its own. Today that is
:mod:`segx.analysis.od_spread` (P7.5) — group the OD table by origin and read how
many destinations each demographic class actually reaches from it.

The separation matters practically. An input analysis runs in seconds with no
OSM download, no snapping and no Dijkstra, so it is the first thing to look at
when a new OD file arrives — and when it says the classes all reach the same
places, the expensive run that follows will not find segregation the table does
not contain.
"""

from segx.analysis.od_spread import (
    ODSpreadResult,
    analyse_od_spread,
    class_destination_overlap,
    class_flows,
    destination_spread,
    od_spread_unavailable_reason,
)

__all__ = [
    "ODSpreadResult",
    "analyse_od_spread",
    "class_destination_overlap",
    "class_flows",
    "destination_spread",
    "od_spread_unavailable_reason",
]
