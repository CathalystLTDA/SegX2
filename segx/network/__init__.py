"""Per-mode routable graphs (walk, drive; 'transit' reserved for 2.2): build,
clean, project to the working CRS, attach mode-specific travel-time weights;
on-disk cache keyed by ``(bbox_hash, osm_snapshot_date, network_type)``; snapping
of origins/POIs/OD endpoints with a distance audit; a mode→graph registry."""
