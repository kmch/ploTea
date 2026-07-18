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


def resolve_crs(name: str) -> ccrs.CRS:
    """
    Return the CRS for a preset name ('equal_earth', 'europe_laea').

    Parameters
    ----------
    name : str
        A key of the CRS preset table.

    Returns
    -------
    cartopy.crs.CRS

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.

    Examples
    --------
    >>> crs = resolve_crs('equal_earth')

    """
    try:
        factory = _CRS_PRESETS[name]
    except KeyError:
        known = ', '.join(sorted(_CRS_PRESETS))
        raise KeyError(f'unknown CRS preset {name!r}; known presets: {known}') from None
    return factory()
