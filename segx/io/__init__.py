"""Input/output boundary: loaders for boundary, demographics, network, POIs and
OD pairs; deliverable writers. Internal artifacts are GeoParquet; deliverables
are GeoPackage (per contract clause 2.5). All loaders reproject incoming geometry
to ``Settings.crs.working_crs`` immediately."""
