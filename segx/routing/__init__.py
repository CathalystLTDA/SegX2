"""Routing: three trajectory-producing strategies feeding one aggregator.

* A — ``nearest_poi``: the single closest POI of each category per origin.
* B — ``poi_catchment``: *all* POIs of each category reachable within a
  configurable cost budget (default 30 min walk @ 5 km/h).
* C — ``od_router``: user-supplied origin→destination pairs (commute, school…).

Default engine is Pandana (``pandana_engine``); NetworkX (``networkx_engine``) is
a validation oracle on small samples. Output is a unified stream of
:class:`segx.routing.trajectories.Trajectory`."""
