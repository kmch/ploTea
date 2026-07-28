"""
Named regions of interest -- bounding boxes a string argument can point to.

Notes
-----
Each entry is ``(minx, miny, maxx, maxy)`` in lon/lat degrees (EPSG:4326), 
i.e. ``(lon_min, lat_min, lon_max, lat_max)``.

 ``Bbox.from_any`` / ``Bbox.from_name`` (in ``maps.vector``)
turn a key of this table into a ``Bbox``.

"""
from plotea.log import get_logger

_log = get_logger(__name__)


# (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max) in degrees.
ROIS = {
    'eu'          : (-10.000, 35.000, 30.000, 72.000),
    'eu_wide'     : (-15.000, 35.000, 35.000, 72.000),
    'iberia'      : (-9.479, 36.025, 4.322, 43.764),    # Spain + Portugal (mainland)
    
    # Country-level ROIs (mainland only)
    'pl'          : (12.128, 47.020, 26.105, 56.838),   # Poland + 2 deg padding
    'uk'          : (-10.390, 50.021, 1.746, 60.831),   # UK + Ireland
    'fr'          : (-4.762, 41.384, 9.556, 51.097),    # France (mainland)
    # Estonia
    'ee'          : (21.764, 57.509, 28.208, 59.822),
    'ee_parnu'    : (24.491, 57.973, 25.958, 59.1308),

    # Sub-national zoom windows
    'greater_london' : (-0.600, 51.250,  0.350, 51.720),
    'poland_central' : (18.500, 51.200, 21.700, 52.900),   # Łódź -- Warsaw
    'po_valley'      : ( 7.500, 44.600, 12.600, 46.000),
}
