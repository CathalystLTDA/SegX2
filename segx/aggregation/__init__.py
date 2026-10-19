"""Streaming per-edge aggregation of demographic mass, partitioned by
``(mode, category, binding)`` so walk/park/nearest, walk/park/catchment,
car/commute/od, … are reportable separately. Optionally also keeps a sampled
trajectory audit log. No per-trajectory storage in the hot path.

``path_accumulator`` (P7.3) runs beside it over the same trajectory stream for the
one measure that is *not* a property of a segment — the fractal dimension of a
path — keeping only per-``(partition, attribute, class)`` sums, so the same
bounded-memory guarantee holds."""
