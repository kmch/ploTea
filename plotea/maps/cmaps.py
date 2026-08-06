"""
Ready-made colormaps for rasters: discrete class schemes and a land-terrain ramp.

The discrete builders return a ``DiscreteCmap`` bundling the matplotlib ``cmap`` + ``norm``
(so a class raster draws with the official colours) together with the class ``values`` and
``labels`` (so a colorbar can be ticked and labelled). ``land_terrain`` returns a continuous
colormap for elevation. plotea does no raster IO -- pass the raster to ``plot_raster`` and
the ``cmap`` / ``norm`` alongside it.

Notes
-----
``esa_worldcover`` uses the official ESA WorldCover class colours; ``koeppen_geiger`` uses
the Beck et al. (2018) standard Köppen-Geiger colours.

"""
from dataclasses import dataclass

from matplotlib.colors import Colormap, Normalize, from_levels_and_colors

__all__ = ['DiscreteCmap', 'esa_worldcover', 'koeppen_geiger', 'land_terrain']


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

    Examples
    --------
    >>> scheme = esa_worldcover()
    >>> im = plot_raster(lulc, extent=ext, ax=ax, cmap=scheme.cmap, norm=scheme.norm)
    >>> cb = fig.colorbar(im); cb.set_ticks(scheme.values); cb.set_ticklabels(scheme.labels)

    """
    cmap: Colormap
    norm: Normalize
    values: list
    labels: list


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
    return DiscreteCmap(cmap, norm, values, labels)


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
    return DiscreteCmap(cmap, norm, values, labels)


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
    >>> plot_raster(dem, extent=ext, ax=ax, cmap=land_terrain())

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap('terrain')
    return LinearSegmentedColormap.from_list('land_terrain', base(np.linspace(0.25, 1.0, 256)))
