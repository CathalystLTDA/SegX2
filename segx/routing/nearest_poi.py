"""Strategy A — for each origin, the single nearest POI of each configured
category (nearest park, nearest bus stop, …). Default mode ``'walk'``. Emits one
:class:`~segx.routing.trajectories.Trajectory` per (origin, category) with
``binding='nearest'``; when no POI of a category is reachable within
``poi.nearest.max_distance_m`` the trajectory has ``path=None`` and a ``reason``
(``unreachable`` / ``none_in_boundary`` / ``over_max_distance``) — never silently
dropped. P3.

Routes on the ``"length"`` (metres) weight so ``max_distance_m`` applies directly.

Origins are routed **a block at a time, one engine call per category**
(``engine.nearest_many``), and emitted in the order they came in — origin by
origin, category by category — exactly as if each had been routed alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from segx.routing.inputs import origin_blocks
from segx.routing.trajectories import Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator, Mapping

    from segx.config import Settings
    from segx.routing.engine import RoutingEngine
    from segx.routing.inputs import RoutableOrigin, RoutablePOI

#: Edge weight nearest-binding distances are measured in (metres).
NEAREST_WEIGHT = "length"
#: Origins routed per engine call — bounds what is held before it is yielded.
ORIGIN_BLOCK = 5_000


def nearest_poi_trajectories(
    origins: Iterable[RoutableOrigin],
    pois_by_category: Mapping[str, list[RoutablePOI]],
    engine: RoutingEngine,
    settings: Settings,
    *,
    mode: str = "walk",
) -> Iterator[Trajectory]:
    """Yield one ``Trajectory`` (``binding='nearest'``) per (origin, category)."""
    cfg = settings.poi.nearest
    if not cfg.enabled:
        return
    categories = settings.poi.categories_for("nearest")
    pois_of = {category: list(pois_by_category.get(category, [])) for category in categories}

    for block in origin_blocks(origins, ORIGIN_BLOCK):
        nodes = [origin.node for origin in block]
        found = {
            category: engine.nearest_many(nodes, pois, mode, weight=NEAREST_WEIGHT)
            for category, pois in pois_of.items()
            if pois
        }
        for index, origin in enumerate(block):
            for category in categories:
                if category not in found:
                    yield Trajectory(
                        origin.origin_id, "", mode, category, "nearest", weight=origin.weight, reason="none_in_boundary"
                    )
                    continue
                result = found[category][index]
                if result is None:
                    yield Trajectory(
                        origin.origin_id, "", mode, category, "nearest", weight=origin.weight, reason="unreachable"
                    )
                    continue
                if cfg.max_distance_m is not None and result.cost > cfg.max_distance_m:
                    yield Trajectory(
                        origin.origin_id,
                        result.target_id,
                        mode,
                        category,
                        "nearest",
                        weight=origin.weight,
                        cost=result.cost,
                        reason="over_max_distance",
                        meta={"poi_node": result.target_node},
                    )
                    continue
                yield Trajectory(
                    origin.origin_id,
                    result.target_id,
                    mode,
                    category,
                    "nearest",
                    weight=origin.weight,
                    path=result.path,
                    cost=result.cost,
                    meta={"poi_node": result.target_node},
                )


__all__ = ["NEAREST_WEIGHT", "ORIGIN_BLOCK", "nearest_poi_trajectories"]
