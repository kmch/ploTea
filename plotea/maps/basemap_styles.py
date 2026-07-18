"""
Basemap styles: fill colours and line weights for the map layers.

Notes
-----
``BasemapStyle`` is a frozen data record; ``BASEMAP_PLAIN`` and ``BASEMAP_MUTED``
are two ready-made instances. Line widths default below one point because at
world scale a one-point coastline reads as a black smear; the publication style
raises them.

"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BasemapStyle:
    """
    Fill colours and line weights for the basemap layers. A frozen data record.

    Parameters
    ----------
    land, ocean : str
        Fill colours for the land and ocean polygons.
    coastline, border : str
        Line colours for coastlines and country borders.
    coastline_width, border_width : float
        Line widths in points.
    graticule : str
        Colour of the lon/lat gridlines.
    graticule_width : float
        Gridline width in points.

    Examples
    --------
    >>> BASEMAP_PLAIN.land
    'white'
    >>> from dataclasses import replace
    >>> thick_borders = replace(BASEMAP_MUTED, border_width=0.8)

    """
    land: str = 'white'
    ocean: str = 'whitesmoke'
    coastline: str = '#333333'
    coastline_width: float = 0.5
    border: str = '#666666'
    border_width: float = 0.4
    graticule: str = 'grey'
    graticule_width: float = 0.4


BASEMAP_PLAIN = BasemapStyle()
BASEMAP_MUTED = BasemapStyle(land='#efe9e1', ocean='#dce6ec', coastline='#5b5b5b', border='#8a8a8a')
