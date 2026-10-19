# Reports

Technical write-ups, one per development phase: what was built, how it was verified, and
what was measured. Phase numbers match the ones used in the code's docstrings (`P3`,
`P7.3`, …).

Each phase report describes the engine **as it was when that phase closed**. Later phases
sometimes changed earlier decisions; where that happened, the earlier report says so at the
top and points to the report that changed it.

| Phase | Report | Topic |
|---|---|---|
| P0–P1 | [P0-P1-foundations.md](P0-P1-foundations.md) | Project skeleton, configuration, CRS handling, OSM networks and cache, loaders, synthetic validation city |
| P2 | [P2-origins.md](P2-origins.md) | Origins and their demographic vectors; LODES-style flow files |
| P3 | [P3-routing.md](P3-routing.md) | The three trajectory strategies (nearest, catchment, OD); Pandana and the NetworkX oracle |
| P4 | [P4-aggregation-diversity.md](P4-aggregation-diversity.md) | Streaming per-segment aggregation; the diversity (entropy) measure |
| P4.5, P5 | [P5-segregation-metrics-route-sets.md](P5-segregation-metrics-route-sets.md) | The five segregation measures; alternative routes per OD pair |
| P7.3 | [P7.3-path-fragmentation.md](P7.3-path-fragmentation.md) | Fractal dimension of trajectories (box counting) and its methodological issues |
| P7.5 | [P7.5-od-destination-spread.md](P7.5-od-destination-spread.md) | Destinations per class in the OD table; rarefaction; Philadelphia results |
| P8 | [P8-output-writers.md](P8-output-writers.md) | `segments.gpkg` + `metrics.xlsx`, provenance stamps, audits |

Reference:

- [metric-formulas.md](metric-formulas.md) — the exact formula, edge-case convention and
  implementing code of every per-segment and per-group metric, with a worked example.

Public transport (P7), the service window and cross-mode exposure (P7.1) and the shared
street space across modes (P7.2) have no standalone report yet; their behaviour is
documented in the code (`segx/network/transit.py`, `segx/network/street_space.py`,
`segx/metrics/cross_mode_exposure.py`) and summarised in
[metric-formulas.md](metric-formulas.md) §5b.
