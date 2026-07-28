"""
A black-and-white segmented scale bar -- the standard map element cartopy lacks.

Draws in the axes' own projected coordinates, so a bar of ``length_km`` renders at
the true on-map length. Intended for a metre-based projection (e.g. LAEA); there
the projected length equals ground distance closely enough for a scale bar.

"""
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


def _nice(value: float) -> float:
    """
    Round up to a cartographer's 1/2/5 x 10^n length.

    Examples
    --------
    >>> _nice(37), _nice(120)
    (50.0, 100.0)

    """
    exp = np.floor(np.log10(value))
    frac = value / 10 ** exp
    step = 1 if frac < 1.5 else 2 if frac < 3.5 else 5 if frac < 7.5 else 10
    return step * 10 ** exp


def scalebar(ax, length_km=None, location: str = 'lower right', segments: int = 4, pad: float = 0.07, height: float = 0.018, fontsize: float = 7, linewidth: float = 0.8):
    """
    Draw a segmented black/white scale bar with distance labels on a projected map axes.

    Parameters
    ----------
    ax : cartopy GeoAxes
        A metre-projection map axes (its x limits are read in metres).
    length_km : float, optional
        Total bar length in km; a nice 1/2/5 value near a quarter of the map width
        is chosen when None.
    location : str
        Corner: 'lower right' (default), 'lower left', 'upper right', 'upper left'.
    segments : int
        Number of alternating black/white blocks.
    pad : float
        Inset from the axes edges, as a fraction of the axes.
    height : float
        Bar thickness, as a fraction of the axes height.
    fontsize : float
        Label size in points.
    linewidth : float
        Block outline width.

    Returns
    -------
    float
        The bar length actually drawn, in km.

    Notes
    -----
    Labels carry a white halo so they read over dark terrain. Everything is drawn at
    a high zorder with ``clip_on=False`` so the bar sits above the map near the edge.

    Examples
    --------
    >>> scalebar(ax, location='lower right')          # auto length
    >>> scalebar(ax, length_km=100, segments=2)

    """
    # Draw in the axes' projected (metre) data coords. The explicit transData is
    # essential on a plotea LonLatAxes: without it the axes assumes lon/lat and the
    # metre coordinates would be read as degrees, placing the bar off the map.
    t = ax.transData
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = x1 - x0, y1 - y0
    if length_km is None:
        length_km = _nice(0.25 * xr / 1000.0)
    total = length_km * 1000.0

    xs = x1 - pad * xr - total if 'right' in location else x0 + pad * xr
    yb = y0 + pad * yr if 'lower' in location else y1 - pad * yr - height * yr
    h = height * yr
    seg = total / segments

    for i in range(segments):
        colour = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle((xs + i * seg, yb), seg, h, facecolor=colour, edgecolor='black', linewidth=linewidth, transform=t, zorder=6, clip_on=False))

    halo = [pe.withStroke(linewidth=2.0, foreground='white')]
    for frac, val in [(0.0, 0.0), (0.5, length_km / 2), (1.0, length_km)]:
        text = f'{val:g}' if frac < 1.0 else f'{val:g} km'
        ax.text(xs + frac * total, yb + h * 1.5, text, ha='center', va='bottom', fontsize=fontsize, transform=t, zorder=6, clip_on=False, path_effects=halo)
    return length_km
