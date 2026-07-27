"""
Colour scales -- classified (discrete-band) colour maps for choropleth-style layers.

``discrete`` builds a banded ``(cmap, norm)`` pair from class boundaries: the
default is four Reds bands split at 2/5/10, the look of an exposure or hazard map.
It is deliberately generic -- the boundaries are passed in as data, so the module
carries no domain thresholds of its own.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap


def _fmt(v: float) -> str:
    """
    Format a class boundary compactly -- integer when whole, else trimmed float.

    Examples
    --------
    >>> _fmt(5.0), _fmt(2.5)
    ('5', '2.5')

    """
    return str(int(v)) if float(v).is_integer() else f'{v:g}'


def discrete(bounds=None, values=None, n: int = 4, cmap: str = 'Reds', low: float = 0.15):
    """
    Build a classified ``(cmap, norm)`` with ``len(bounds) + 1`` colour bands.

    Pass explicit class boundaries with ``bounds``, or let them be derived from a
    data sample with ``values`` (``n`` equal-count quantile bands). With neither,
    the default is four Reds bands split at 2, 5 and 10.

    Parameters
    ----------
    bounds : sequence of float, optional
        The class boundaries (strictly increasing). ``k`` boundaries give ``k + 1``
        bands: below the first, between each pair, and above the last.
    values : array-like, optional
        A data sample; when ``bounds`` is None the boundaries become the interior
        ``n``-quantiles of these values.
    n : int
        Number of bands to derive from ``values`` (ignored when ``bounds`` given).
    cmap : str
        Any matplotlib colormap name to sample the band colours from (e.g. 'Reds').
    low : float
        Where in the colormap the lightest band starts (0-1); raising it drops the
        palest tints so the first band still reads as colour, not white.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm

    Notes
    -----
    ``k`` boundaries give ``k + 1`` bands: the interior ``k - 1`` are the listed
    colours, and the open-ended below-first and above-last bands are carried by the
    colormap's ``set_under``/``set_over`` colours. Pass ``extend='both'`` to
    ``colorbar`` to draw those two as end triangles.

    Examples
    --------
    >>> cmap, norm = discrete()                       # 4 Reds bands at 2/5/10
    >>> cmap, norm = discrete(bounds=[1, 25, 50, 100])
    >>> cmap, norm = discrete(values=df.obs_value, n=5, cmap='YlOrRd')
    >>> ax.scatter(x, y, c=v, cmap=cmap, norm=norm)

    """
    if bounds is None:
        if values is not None:
            qs = np.linspace(0.0, 1.0, n + 1)[1:-1]
            bounds = list(np.quantile(np.asarray(values, dtype=float), qs))
        else:
            bounds = [2.0, 5.0, 10.0]
    bounds = [float(b) for b in bounds]
    if len(bounds) < 2:
        raise ValueError('discrete needs at least 2 boundaries (3 bands); got %r' % (bounds,))
    colours = plt.get_cmap(cmap)(np.linspace(low, 1.0, len(bounds) + 1))
    # interior bands are the listed colours; the open-ended end bands are under/over.
    cmap_d = ListedColormap(colours[1:-1]).with_extremes(under=colours[0], over=colours[-1])
    norm = BoundaryNorm(bounds, ncolors=cmap_d.N)
    return cmap_d, norm


def discrete_colorbar(fig, cax, cmap, norm, label=None, orientation: str = 'vertical', **kwargs):
    """
    Draw the legend for a ``discrete`` scale as equal-height blocks (no end triangles).

    A ``discrete`` scale carries its open-ended first and last bands as the
    colormap's under/over colours, so a plain colorbar renders them as thin
    triangles. This instead lays out every band -- including those two -- as a
    block of equal size, with the class boundaries ticked between blocks, which is
    the classified-legend look of an exposure or hazard map.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure the colorbar belongs to.
    cax : matplotlib.axes.Axes
        The axes to draw the colorbar into (e.g. an ``ax.inset_axes([...])``).
    cmap, norm : from ``discrete``
        The classified colormap and boundary norm.
    label : str, optional
        Axis label for the colorbar.
    orientation : str
        'vertical' (default) or 'horizontal'.
    **kwargs
        Passed to ``fig.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar

    Examples
    --------
    >>> cmap, norm = discrete()
    >>> cax = ax.inset_axes([0.9, 0.55, 0.03, 0.4])
    >>> cb = discrete_colorbar(fig, cax, cmap, norm, label='TN (mg L$^{-1}$)')

    """
    colours = [cmap.get_under()] + [cmap(i) for i in range(cmap.N)] + [cmap.get_over()]
    n = len(colours)
    block_cmap = ListedColormap(colours)
    block_norm = BoundaryNorm(list(range(n + 1)), n)   # equal integer-wide blocks
    cb = fig.colorbar(ScalarMappable(cmap=block_cmap, norm=block_norm), cax=cax, orientation=orientation, spacing='uniform', **kwargs)
    cb.set_ticks(range(1, n))                            # ticks sit between blocks
    cb.set_ticklabels([_fmt(b) for b in norm.boundaries])
    if label is not None:
        cb.set_label(label)
    return cb
