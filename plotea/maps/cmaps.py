"""
Ready-made colormaps for rasters: discrete class schemes and a land-terrain ramp.

The discrete builders return a ``DiscreteCmap`` bundling the matplotlib ``cmap`` + ``norm``
(so a class raster draws with the official colours) together with the class ``values`` and
``labels`` (so a colorbar can be ticked and labelled). ``land_terrain`` returns a continuous
colormap for elevation. plotea does no raster IO -- pass the raster to ``RasterPlotter.imshow`` and
the ``cmap`` / ``norm`` alongside it.

Notes
-----
``esa_worldcover`` uses the official ESA WorldCover class colours; ``koeppen_geiger`` uses
the Beck et al. (2018) standard Köppen-Geiger colours.

"""
from dataclasses import dataclass

from matplotlib.colors import Colormap, Normalize, from_levels_and_colors

__all__ = ['ContinuousCmap', 'DiscreteCmap', 'bdod', 'binary_mask', 'clay', 'elevation',
           'elevation_cm', 'esa_worldcover', 'koeppen_geiger', 'land_terrain', 'livestock',
           'population',
           'precipitation', 'precipitation_annual', 'precipitation_driest_quarter',
           'precipitation_wettest_quarter', 'ruggedness', 'sand', 'seasonality',
           'sediment_thickness',
           'slope', 'soil_carbon', 'soil_ph', 'temperature', 'temperature_range',
           'wetness_index']


@dataclass(frozen=True)
class DiscreteCmap:
    """
    A discrete colour scheme for a classified raster: cmap + norm + class values/labels.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Listed colormap, one colour per class.
    norm : matplotlib.colors.Normalize
        Boundary norm mapping class values to their colours.
    values : list of int
        The class codes, in colormap order -- use as colorbar ticks.
    labels : list of str
        Human-readable class names matching ``values`` -- use as tick labels.
    label : str, optional
        What the classification *is*, for the bar's own label: the class names say which
        class, and this says which scheme they belong to.

    Examples
    --------
    >>> scheme = esa_worldcover()
    >>> im = RasterPlotter.imshow(lulc, extent=ext, ax=ax, cmap=scheme.cmap, norm=scheme.norm)
    >>> cb = fig.colorbar(im); cb.set_ticks(scheme.values); cb.set_ticklabels(scheme.labels)

    """
    cmap: Colormap
    norm: Normalize
    values: list
    labels: list
    label: str = None


def esa_worldcover() -> DiscreteCmap:
    """
    ESA WorldCover land-cover colour scheme (official class colours).

    Returns
    -------
    DiscreteCmap
        The 11 land-cover classes (Tree .. Moss) with their official colours.

    Examples
    --------
    >>> esa_worldcover().labels[0]
    'Tree'

    """
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    colors = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4',
              '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']
    labels = ['Tree', 'Shrub', 'Grass', 'Crop', 'Built', 'Bare', 'Snow/ice',
              'Water', 'Wetland', 'Mangrove', 'Moss']
    levels = [5, 15, 25, 35, 45, 55, 65, 75, 85, 92.5, 97.5, 102.5]   # class-centred boundaries
    cmap, norm = from_levels_and_colors(levels, colors)
    return DiscreteCmap(cmap, norm, values, labels, label='Land cover')


def koeppen_geiger() -> DiscreteCmap:
    """
    Köppen-Geiger climate-classification colour scheme (Beck et al. 2018 standard colours).

    Returns
    -------
    DiscreteCmap
        The 30 climate classes (Af .. EF) with the Beck et al. (2018) standard colours.

    Examples
    --------
    >>> koeppen_geiger().labels[14]
    'Cfb'

    """
    labels = ['Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc',
              'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 'Dsa', 'Dsb', 'Dsc', 'Dsd',
              'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF']
    rgb = [(0, 0, 255), (0, 120, 255), (70, 170, 250), (255, 0, 0), (255, 150, 150),
           (245, 165, 0), (255, 220, 100), (255, 255, 0), (200, 200, 0), (150, 150, 0),
           (150, 255, 150), (100, 200, 100), (50, 150, 50), (200, 255, 80), (100, 255, 80),
           (50, 200, 0), (255, 0, 255), (200, 0, 200), (150, 50, 150), (150, 100, 150),
           (170, 175, 255), (90, 120, 220), (75, 80, 180), (50, 0, 135), (0, 255, 255),
           (55, 200, 255), (0, 125, 125), (0, 70, 95), (178, 178, 178), (102, 102, 102)]
    colors = [(r / 255, g / 255, b / 255) for r, g, b in rgb]
    values = list(range(1, 31))
    levels = [c - 0.5 for c in values] + [30.5]
    cmap, norm = from_levels_and_colors(levels, colors)
    return DiscreteCmap(cmap, norm, values, labels, label='Koppen-Geiger class')


def land_terrain():
    """
    The matplotlib ``terrain`` colormap with its blue underwater low end trimmed off.

    For plotting land elevation, so lowlands read green rather than the bathymetry blue that
    ``terrain`` puts at its bottom. A continuous ``Colormap`` (not a ``DiscreteCmap``).

    Returns
    -------
    matplotlib.colors.Colormap

    Examples
    --------
    >>> RasterPlotter.imshow(dem, extent=ext, ax=ax, cmap=land_terrain())

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap('terrain')
    return LinearSegmentedColormap.from_list('land_terrain', base(np.linspace(0.25, 1.0, 256)))


@dataclass(frozen=True)
class ContinuousCmap:
    """
    A measured quantity's colour scheme: a colormap, and how its values map onto it.

    The continuous twin of :class:`DiscreteCmap`. A colormap alone is not a scheme -- soil pH
    needs its neutral point in the middle of the bar and population needs a logarithmic one,
    and a caller passing ``cmap='RdYlBu'`` has nowhere to say so.

    ``center`` and ``log`` are kept as *intentions* rather than as a built ``Normalize``,
    because a norm has to know the data. ``TwoSlopeNorm(vcenter=0)`` raises on a raster whose
    values never reach zero, which is an ordinary thing for a window of one island to do, so
    the norm is built per raster and quietly falls back to a linear one where the data does
    not straddle the centre. Pass ``norm`` to override all of that.

    Parameters
    ----------
    cmap : str or Colormap
    center : float, optional
        The value to put at the middle of the bar: 0 for a temperature in Celsius, 70 for a
        pH stored as pH x 10.
    log : bool, optional
        Logarithmic, for a quantity spanning orders of magnitude. Zero and negative values
        are kept rather than dropped, by way of a symmetric log scale.
    norm : matplotlib.colors.Normalize, optional
        An explicit norm, which overrides ``center`` and ``log``.
    label : str, optional
        What the quantity is, for the colorbar.
    unit : str, optional
        The unit the bar is labelled in, *after* ``scale`` -- so metres for an elevation
        stored in centimetres and drawn with ``scale=0.01``. Where the stored unit is kept,
        the label says so: ``'pH x 10'``.
    scale : float, optional
        Multiply the values by this before drawing, GDAL's own ``physical = raw * scale``.
        **Only for a band that declares no scale of its own**: a declared one is already
        applied by the read, and setting both multiplies twice -- a class proportion stored
        as int16 with ``scale=0.0001`` comes back as 0 to 1 and a second 0.0001 makes the
        whole map read as zero. This is for the case nothing declares, such as an elevation
        in centimetres whose band says nothing, where it is better than a bar reading 181703
        for a mountain 1817 m high.

    Examples
    --------
    >>> soil_ph()
    ContinuousCmap(cmap='RdYlBu', center=70, ...)
    >>> raster.plot(cmap=soil_ph())

    """

    cmap: object
    center: float = None
    log: bool = False
    norm: Normalize = None
    label: str = None
    unit: str = None
    scale: float = 1.0

    def norm_for(self, values):
        """
        The norm to draw these values with: explicit, diverging, logarithmic or none.

        Parameters
        ----------
        values : numpy.ndarray or numpy.ma.MaskedArray
            The data about to be drawn; only its finite extremes are used.

        Returns
        -------
        matplotlib.colors.Normalize or None
            None means "linear, limits from the data", which is matplotlib's own default.

        Examples
        --------
        >>> soil_ph().norm_for(values)

        """
        import numpy as np
        from matplotlib.colors import SymLogNorm, TwoSlopeNorm

        if self.norm is not None:
            return self.norm
        finite = np.asarray(values[np.isfinite(values)] if not np.ma.isMaskedArray(values)
                            else values.compressed())
        if finite.size == 0:
            return None
        low, high = float(finite.min()), float(finite.max())
        if self.center is not None:
            # Only where the data actually spans the centre. Otherwise a diverging bar would
            # show one half of itself, which says "all of this is on one side" far less
            # clearly than a plain ramp does.
            if low < self.center < high:
                return TwoSlopeNorm(vcenter=self.center, vmin=low, vmax=high)
            return None
        if self.log:
            # Symmetric rather than plain log: these counts include zeros, and LogNorm masks
            # every one of them, turning "nobody lives here" into "no data".
            return SymLogNorm(linthresh=1, vmin=low, vmax=high)
        return None


def _quantity(cmap, label, unit=None, center=None, log=False, scale=1.0) -> ContinuousCmap:
    """
    One measured quantity's scheme, so each named builder below is its own one-liner.

    Examples
    --------
    >>> _quantity('Oranges', 'Clay content', 'g/kg')

    """
    return ContinuousCmap(cmap=cmap, center=center, log=log, label=label, unit=unit, scale=scale)


def bdod() -> ContinuousCmap:
    """
    Soil bulk density: dense soil dark, light soil pale.

    SoilGrids stores it in centigrams per cubic centimetre, so 130 is 1.3 g/cm3.

    Examples
    --------
    >>> raster.plot(cmap=bdod())

    """
    return _quantity('copper_r', 'Bulk density', 'cg/cm3')


def binary_mask(label=None) -> DiscreteCmap:
    """
    A two-class mask: absent pale, present green.

    Parameters
    ----------
    label : str, optional
        What the mask is of, for the bar -- "Tree cover", "Stream network".

    Examples
    --------
    >>> raster.plot(cmap=binary_mask('Tree cover'))

    """
    cmap, norm = from_levels_and_colors([-0.5, 0.5, 1.5],
                                        [(0.93, 0.93, 0.93), (0.18, 0.49, 0.20)])
    return DiscreteCmap(cmap, norm, [0, 1], ['Absent', 'Present'], label=label)


def clay() -> ContinuousCmap:
    """
    Soil clay content, in SoilGrids' grams per kilogram -- 470 is 47 %.

    Examples
    --------
    >>> raster.plot(cmap=clay())

    """
    return _quantity('Oranges', 'Clay content', 'g/kg')


def elevation(unit='m') -> ContinuousCmap:
    """
    Land elevation on the terrain ramp, lowlands green and summits white.

    Parameters
    ----------
    unit : str, optional
        What the values are stored in, for the bar. Metres by default -- but see
        :func:`elevation_cm`, because that default is wrong more often than it looks.

    Examples
    --------
    >>> raster.plot(cmap=elevation())

    """
    return _quantity(land_terrain(), 'Elevation', unit)


def elevation_cm() -> ContinuousCmap:
    """
    Land elevation stored in **centimetres**, drawn in metres.

    A separate builder rather than an argument, because the unit is the whole point: the same
    island ships twice, once as a float32 DEM in metres reaching 1819 and once as an int32
    covariate in centimetres reaching 181703, and nothing on either band declares a scale. So
    the hundred lives here, and both come out as metres on the same bar.

    Examples
    --------
    >>> elevation_cm().scale
    0.01

    """
    return _quantity(land_terrain(), 'Elevation', 'm', scale=0.01)


def livestock() -> ContinuousCmap:
    """
    Livestock density, logarithmic: a few herds carry most of the range.

    Examples
    --------
    >>> raster.plot(cmap=livestock())

    """
    return _quantity('YlOrRd', 'Livestock density', log=True)


def population() -> ContinuousCmap:
    """
    People per cell, logarithmic and keeping the empty cells.

    A city is four orders of magnitude above a hamlet, so a linear bar shows the city and
    nothing else. Zeros survive the log, because "nobody lives here" is a measurement.

    Examples
    --------
    >>> raster.plot(cmap=population())

    """
    return _quantity('inferno', 'Population', 'people per cell', log=True)


def precipitation(label='Precipitation') -> ContinuousCmap:
    """
    Precipitation in millimetres: dry pale, wet deep blue.

    Parameters
    ----------
    label : str, optional
        Which precipitation. The WorldClim bioclimatic set holds several, and they share a
        ramp so they can be compared against each other -- but not a caption, since a map of
        the wettest quarter and one of the driest are opposite halves of the year.

    Examples
    --------
    >>> raster.plot(cmap=precipitation())

    """
    return _quantity('YlGnBu', label, 'mm')


def precipitation_annual() -> ContinuousCmap:
    """
    WorldClim bio12: total precipitation over the year.

    Examples
    --------
    >>> precipitation_annual().label
    'Annual precipitation'

    """
    return precipitation('Annual precipitation')


def precipitation_driest_quarter() -> ContinuousCmap:
    """
    WorldClim bio17: precipitation of the driest quarter.

    Examples
    --------
    >>> precipitation_driest_quarter().label
    'Precipitation of driest quarter'

    """
    return precipitation('Precipitation of driest quarter')


def precipitation_wettest_quarter() -> ContinuousCmap:
    """
    WorldClim bio16: precipitation of the wettest quarter.

    Examples
    --------
    >>> precipitation_wettest_quarter().label
    'Precipitation of wettest quarter'

    """
    return precipitation('Precipitation of wettest quarter')


def ruggedness() -> ContinuousCmap:
    """
    Terrain ruggedness index, flat pale and broken dark.

    Examples
    --------
    >>> raster.plot(cmap=ruggedness())

    """
    return _quantity('GnBu', 'Terrain ruggedness', 'index x 100')


def sand() -> ContinuousCmap:
    """
    Soil sand content, in grams per kilogram.

    Examples
    --------
    >>> raster.plot(cmap=sand())

    """
    return _quantity('YlOrBr', 'Sand content', 'g/kg')


def seasonality() -> ContinuousCmap:
    """
    Precipitation seasonality, the coefficient of variation across the months.

    Examples
    --------
    >>> raster.plot(cmap=seasonality())

    """
    return _quantity('viridis', 'Precipitation seasonality', 'CV')


def sediment_thickness(label='Sediment thickness') -> ContinuousCmap:
    """
    Soil and sedimentary deposit thickness in metres, thin pale and thick dark.

    Parameters
    ----------
    label : str, optional
        Which thickness. A dataset may hold several -- an average over all terrain and one
        restricted to valley bottoms -- and they share a ramp so they can be compared, but
        two maps captioned alike are two maps nobody can tell apart.

    Examples
    --------
    >>> raster.plot(cmap=sediment_thickness('Average sediment thickness'))

    """
    return _quantity('pink_r', label, 'm')


def slope() -> ContinuousCmap:
    """
    Terrain slope, gentle pale and steep red.

    Examples
    --------
    >>> raster.plot(cmap=slope())

    """
    return _quantity('YlOrRd', 'Slope', 'degrees x 100')


def soil_carbon() -> ContinuousCmap:
    """
    Soil organic carbon, logarithmic: peat is an order of magnitude above mineral soil.

    Examples
    --------
    >>> raster.plot(cmap=soil_carbon())

    """
    return _quantity('YlGn', 'Soil organic carbon', 'dg/kg', log=True)


def soil_ph() -> ContinuousCmap:
    """
    Soil pH, diverging about neutral -- acid red, alkaline blue.

    SoilGrids stores pH multiplied by ten, so neutral is 70 and the centre is set there. On a
    window that is acid throughout, the scheme falls back to a plain ramp rather than showing
    half a bar.

    Examples
    --------
    >>> raster.plot(cmap=soil_ph())

    """
    return _quantity('RdYlBu', 'Soil pH', 'pH x 10', center=70)


def temperature() -> ContinuousCmap:
    """
    Air temperature in Celsius, diverging about freezing.

    Examples
    --------
    >>> raster.plot(cmap=temperature())

    """
    return _quantity('RdBu_r', 'Mean annual temperature', 'C', center=0)


def temperature_range() -> ContinuousCmap:
    """
    Annual temperature range: an interval, so sequential rather than diverging.

    Examples
    --------
    >>> raster.plot(cmap=temperature_range())

    """
    return _quantity('magma', 'Annual temperature range', 'C')


def wetness_index() -> ContinuousCmap:
    """
    Topographic wetness index, dry pale and converging purple.

    Examples
    --------
    >>> raster.plot(cmap=wetness_index())

    """
    return _quantity('BuPu', 'Topographic wetness index', 'index x 100')
