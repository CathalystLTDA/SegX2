"""Strategy B — for each origin, *all* POIs of each configured category reachable
within the configured cost budget (default 30 min walk @ 5 km/h; a distance
budget and per-mode budgets are also supported). Emits one
:class:`~segx.routing.trajectories.Trajectory` per (origin, reachable POI) with
``binding='catchment'``. Trajectory weight follows ``poi.catchment.weight_mode``:
``'unit'`` → ``origin.weight`` each (every reachable POI is a real opportunity);
``'inverse_count'`` → ``origin.weight / k`` over the ``k`` reachable POIs of that
category. An empty catchment is a *real* zero (the resident reaches no POI of that
category in budget) — it emits no trajectory and is recorded as a count by the
orchestrator, never errored or fabricated. P3.

When both a time and a distance budget apply, a POI must satisfy **both** (the
more restrictive wins); the reported ``cost``/``path`` come from the time budget
when present, else the distance budget.

Origins are routed **a block at a time, one engine call per category and
constraint** (``engine.reachable_within_many``), and emitted in the order they
came in. A catchment is the binding that multiplies trajectories — every origin
times every POI in budget — so the block also bounds how many routes exist
before they are yielded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from segx.routing.inputs import origin_blocks
from segx.routing.trajectories import Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator, Mapping

    from segx.config import CatchmentBindingConfig, Settings
    from segx.routing.engine import RouteResult, RoutingEngine
    from segx.routing.inputs import RoutableOrigin, RoutablePOI


#: Origins routed per engine call. Smaller than the nearest binding's block: here
#: every origin yields a route to *each* POI in budget, so this is what bounds the
#: routes held at once.
ORIGIN_BLOCK = 500


def resolve_budget_constraints(cfg: CatchmentBindingConfig, mode: str) -> list[tuple[str, float]]:
    """The ``(weight, budget)`` pairs a POI must satisfy, primary (time) first.

    ``budget_minutes`` (overridable per mode) measures the ``travel_time_min``
    weight; ``distance_budget_m`` measures ``length``. Either or both may apply.
    """
    constraints: list[tuple[str, float]] = []
    minutes = cfg.per_mode_budget_minutes.get(mode, cfg.budget_minutes)
    if minutes is not None:
        constraints.append(("travel_time_min", float(minutes)))
    if cfg.distance_budget_m is not None:
        constraints.append(("length", float(cfg.distance_budget_m)))
    return constraints


def _reachable_all_constraints(
    engine: RoutingEngine,
    origin_nodes: list[int],
    pois: list[RoutablePOI],
    mode: str,
    constraints: list[tuple[str, float]],
) -> list[list[RouteResult]]:
    """Per origin node, the POIs reachable under *every* constraint; cost/path from
    the primary one. Secondary constraints only filter, so they skip the paths."""
    if not constraints:
        return [[] for _ in origin_nodes]
    weight, budget = constraints[0]
    primary = engine.reachable_within_many(origin_nodes, pois, mode, weight=weight, budget=budget)
    if len(constraints) == 1:
        return primary
    survivors = [{r.target_id for r in results} for results in primary]
    for weight, budget in constraints[1:]:
        reachable = engine.reachable_within_many(
            origin_nodes, pois, mode, weight=weight, budget=budget, with_paths=False
        )
        for kept, results in zip(survivors, reachable, strict=True):
            kept &= {r.target_id for r in results}
    return [[r for r in results if r.target_id in kept] for results, kept in zip(primary, survivors, strict=True)]


def catchment_trajectories(
    origins: Iterable[RoutableOrigin],
    pois_by_category: Mapping[str, list[RoutablePOI]],
    engine: RoutingEngine,
    settings: Settings,
    *,
    mode: str = "walk",
) -> Iterator[Trajectory]:
    """Yield ``Trajectory`` objects with ``binding='catchment'``."""
    cfg = settings.poi.catchment
    if not cfg.enabled:
        return
    constraints = resolve_budget_constraints(cfg, mode)
    categories = settings.poi.categories_for("catchment")

    pois_of = {category: list(pois_by_category.get(category, [])) for category in categories}
    for block in origin_blocks(origins, ORIGIN_BLOCK):
        nodes = [origin.node for origin in block]
        found = {
            category: _reachable_all_constraints(engine, nodes, pois, mode, constraints)
            for category, pois in pois_of.items()
            if pois  # no POI of this category in the boundary — a real zero
        }
        for index, origin in enumerate(block):
            for category in categories:
                if category not in found:
                    continue
                results = found[category][index]
                k = len(results)
                if k == 0:
                    continue  # empty catchment within budget — a real zero, not emitted
                weight = origin.weight if cfg.weight_mode == "unit" else origin.weight / k
                for result in results:
                    yield Trajectory(
                        origin.origin_id,
                        result.target_id,
                        mode,
                        category,
                        "catchment",
                        weight=weight,
                        path=result.path,
                        cost=result.cost,
                        meta={"poi_node": result.target_node, "reachable_count": k},
                    )


__all__ = ["catchment_trajectories", "resolve_budget_constraints"]
