"""
Basemap styles: fill colours and line weights for the map layers.

In the ``BaseMap`` flow this module is the ``style`` resolver: ``BasemapStyle.from_any``
turns a style name ('plain', 'muted'), an existing ``BasemapStyle`` or None into the
record ``draw_basemap`` reads its colours and line widths from.

Notes
-----
``BasemapStyle`` is a frozen data record; ``BASEMAP_STYLE_DEFAULT``
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
    >>> BASEMAP_STYLE_DEFAULT.land
    'white'
    >>> from dataclasses import replace
    >>> thick_borders = replace(BASEMAP_STYLE_DEFAULT, border_width=0.8)

    """
    land: str = 'white'
    ocean: str = 'whitesmoke'
    coastline: str = '#333333'
    coastline_width: float = 0.5
    border: str = '#666666'
    border_width: float = 0.4
    graticule: str = 'grey'
    graticule_width: float = 0.4

    @classmethod
    def from_any(cls, style: 'str | BasemapStyle | None') -> 'BasemapStyle':
        """
        Coerce a style name, a ``BasemapStyle`` instance, or None into a ``BasemapStyle``.

        Parameters
        ----------
        style : str or BasemapStyle or None
            A key of the style table ('plain', 'muted'); an existing
            ``BasemapStyle``, returned unchanged; or None for the plain default.

        Returns
        -------
        BasemapStyle

        Raises
        ------
        KeyError
            If ``style`` is an unknown style name.
        TypeError
            If ``style`` is not a string, a ``BasemapStyle``, or None.

        Examples
        --------
        >>> BasemapStyle.from_any('default') is BASEMAP_STYLE_DEFAULT
        True
        >>> BasemapStyle.from_any(None) is BASEMAP_STYLE_DEFAULT
        True

        """
        if style is None:
            return BASEMAP_STYLE_DEFAULT
        if isinstance(style, BasemapStyle):
            return style
        if isinstance(style, str):
            try:
                return _STYLES[style]
            except KeyError:
                known = ', '.join(sorted(_STYLES))
                raise KeyError(f'unknown style {style!r}; known styles: {known}') from None
        raise TypeError(f'unknown style type {type(style).__name__}')


# BASEMAP_PLAIN = BasemapStyle()
# BASEMAP_MUTED = BasemapStyle(land='#efe9e1', ocean='#dce6ec', coastline='#5b5b5b', border='#8a8a8a')
# # Grey land + pale-blue ocean + thin grey borders -- the neutral topographic-map
# # backdrop used by exposure/hazard figures, and the base a hillshade sits on.
# BASEMAP_GREY = BasemapStyle(land='#d9d9d9', ocean='#cfe1f2', coastline='#8c8c8c', coastline_width=0.4, border='#a6a6a6', border_width=0.3, graticule='#b0b0b0')

BASEMAP_STYLE_DEFAULT = BasemapStyle(
    land='darkgrey', ocean='lightblue', 
    coastline='#8c8c8c', coastline_width=1, 
    border='white', border_width=1, 
    graticule='white', graticule_width=1
)


_STYLES = {'default': BASEMAP_STYLE_DEFAULT}
