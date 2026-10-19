"""Origin construction and demographic attribution.

Origin source is conditional: if OD pairs are
supplied, origins are exactly the distinct OD origins (nothing synthesised); if
not, origins are the centroids of the demographic input polygons. Each origin
then receives a demographic vector per ``settings.demographics``."""
