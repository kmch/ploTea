"""
Panel labels -- the bold ``a``, ``b``, ``c`` tags that identify subplots in a figure.

"""


def panel_label(ax, text, loc: str = 'upper left', pad: float = 0.03, fontsize: float = 11, weight: str = 'bold', box: bool = True, **kwargs):
    """
    Place a bold panel tag (e.g. 'a') in a corner of ``ax``, in axes coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel to label.
    text : str
        The tag, e.g. 'a'.
    loc : str
        Corner: 'upper left', 'upper right', 'lower left' or 'lower right'.
    pad : float
        Inset from the corner, as a fraction of the axes.
    fontsize : float
        Tag font size in points.
    weight : str
        Font weight; bold by default.
    box : bool
        Draw a faint white background box behind the tag for legibility on busy maps.
    **kwargs
        Passed to ``ax.text``.

    Returns
    -------
    matplotlib.text.Text

    Notes
    -----
    Positioned via ``transform=ax.transAxes``; on a plotea ``LonLatAxes`` that
    explicit transform is what keeps the tag in the corner rather than being read
    as a lon/lat point.

    Examples
    --------
    >>> panel_label(ax, 'a')
    >>> panel_label(ax_zoom, 'b', loc='upper right')

    """
    x = pad if 'left' in loc else 1.0 - pad
    y = 1.0 - pad if 'upper' in loc else pad
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    bbox = dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7) if box else None
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, fontweight=weight, bbox=bbox, zorder=6, **kwargs)
