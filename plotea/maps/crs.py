"""
Coordinate reference systems (projections). The only place plotea names cartopy CRSs.

Notes
-----
A LAEA world map would be a disc whose antipode smears round the rim, which is
why the world default is Equal Earth and LAEA is reserved for regional presets.

- ``equal_earth`` is plotea's whole-world default: an equal-area projection, so
every country covers screen area proportional to its real area (no inflated
Greenland). 

- ``laea`` is Lambert Azimuthal Equal-Area, excellent for a single
region because it is centred on that region; 

- ``europe_laea`` is the Europe preset
(central_longitude=10, central_latitude=52).



"""
import cartopy.crs as ccrs


def equal_earth(central_longitude: float = 0.0) -> ccrs.EqualEarth:
    """
    Equal Earth projection -- plotea's whole-world default.

    Parameters
    ----------
    central_longitude : float
        Longitude at the centre of the map, in degrees.

    Returns
    -------
    cartopy.crs.EqualEarth

    Examples
    --------
    >>> crs = equal_earth()
    >>> crs = equal_earth(central_longitude=10)

    """
    return ccrs.EqualEarth(central_longitude=central_longitude)


def laea(lon: float, lat: float) -> ccrs.LambertAzimuthalEqualArea:
    """
    Lambert Azimuthal Equal-Area projection centred on (lon, lat).

    Parameters
    ----------
    lon, lat : float
        Longitude and latitude of the projection centre, in degrees.

    Returns
    -------
    cartopy.crs.LambertAzimuthalEqualArea

    Examples
    --------
    >>> crs = laea(10, 52)

    """
    return ccrs.LambertAzimuthalEqualArea(central_longitude=lon, central_latitude=lat)


def europe_laea() -> ccrs.LambertAzimuthalEqualArea:
    """
    Lambert Azimuthal Equal-Area centred on Europe (10 E, 52 N).

    Returns
    -------
    cartopy.crs.LambertAzimuthalEqualArea

    Examples
    --------
    >>> crs = europe_laea()

    """
    return laea(10, 52)


_CRS_PRESETS = {'equal_earth': equal_earth, 'europe_laea': europe_laea}


class Crs:
    """
    Factory namespace: ``Crs.from_any(...)`` coerces loose input into a cartopy CRS.

    Notes
    -----
    ``Crs`` is never instantiated -- it only groups the ``from_any`` coercion with
    the projection presets, so every plotea value type is reached through the same
    ``from_any`` verb. It deliberately returns a bare cartopy ``ccrs.CRS`` rather
    than a plotea wrapper: the axes projection, ``transform=`` and ``to_crs`` all
    speak cartopy/pyproj CRSs directly, so wrapping would only force unwrapping at
    every call site. This is the one ``from_any`` whose return type is foreign.

    Examples
    --------
    >>> Crs.from_any('europe_laea')
    >>> Crs.from_any(None)              # -> Equal Earth, the world default
    >>> Crs.from_any(equal_earth())    # a live CRS passes through unchanged

    """

    @classmethod
    def from_any(cls, crs: str | ccrs.CRS | None) -> ccrs.CRS:
        """
        Coerce a preset name, a live cartopy CRS, or None into a cartopy CRS.

        Parameters
        ----------
        crs : str or cartopy.crs.CRS or None
            A key of the preset table ('equal_earth', 'europe_laea'); an existing
            cartopy CRS, returned unchanged; or None for the Equal Earth default.

        Returns
        -------
        cartopy.crs.CRS

        Raises
        ------
        KeyError
            If ``crs`` is an unknown preset name.
        TypeError
            If ``crs`` is not a string, a cartopy CRS, or None.

        Examples
        --------
        >>> Crs.from_any('equal_earth')

        """
        if crs is None:
            return equal_earth()
        if isinstance(crs, ccrs.CRS):
            return crs
        if isinstance(crs, str):
            try:
                factory = _CRS_PRESETS[crs]
            except KeyError:
                known = ', '.join(sorted(_CRS_PRESETS))
                raise KeyError(f'unknown CRS preset {crs!r}; known presets: {known}') from None
            return factory()
        raise TypeError(f'unknown CRS type {type(crs).__name__}')
