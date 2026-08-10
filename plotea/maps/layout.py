"""
Panel mosaics -- arrange several equal-aspect map axes into aligned rows.

``panel_mosaic`` lays out a large ``main`` overview beside stacked rows of zoom
panels (the podgorski-and-berg arrangement): every row spans the same width, the
panels within a row share a height, and rows may differ in height. Because a map
axes is locked to equal scaling, its height follows its projected aspect ratio,
not the cell -- so the column widths within a row are made proportional to those
aspects, and each row's height falls out of justifying that row to the common
width. Returns bare ``LonLatAxes``; you draw the basemap and data into them.

"""
from plotea.log import get_logger
from plotea.maps import carto
from plotea.maps.vector import Bbox

_log = get_logger(__name__)


def _aspect(bbox, crs) -> float:
    """
    Projected width/height of a resolved ``bbox`` -- how tall the panel renders.

    Examples
    --------
    >>> from plotea.maps.crs import laea_eu
    >>> a = _aspect('fr', laea_eu())

    """
    return carto.projected_aspect(Bbox.from_any(bbox).extent, crs)


def panel_mosaic(fig, crs, main, rows, width: float = 16.0, gap: float = 0.015, margins=(0.05, 0.05, 0.03, 0.01)):
    """
    Lay out a ``main`` overview beside justified ``rows`` of zoom panels; return the axes.

    Every row spans the same width, panels within a row share a height, rows may
    differ in height, and the overview's top edge aligns exactly with the top of
    the upper zoom row. Panels are placed as explicit aspect-matched rectangles
    (not GridSpec cells) so equal-aspect axes cannot drift out of alignment.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to build the mosaic in. Its size is set to the layout's natural
        aspect so the equal-aspect panels tile exactly.
    crs : cartopy.crs.CRS
        The projection every panel is drawn in; sets the panel aspect ratios.
    main : str or list or Bbox
        The bbox of the large overview panel on the left (anything ``Bbox.from_any``
        accepts, e.g. 'eu').
    rows : list of list of (str or list or Bbox)
        Each inner list is one right-hand row of zoom bboxes, laid out left to right.
    width : float
        Target figure width in inches; the height follows from the layout.
    gap : float
        Gap between panels, in units of the right-hand block width.
    margins : tuple of float
        ``(left, bottom, top, right)`` figure-fraction margins; the left/bottom
        margins leave room for the overview's gridline labels.

    Returns
    -------
    ax_main : LonLatAxes
        The overview axes.
    zoom_axes : list of list of LonLatAxes
        The zoom axes, matching the shape of ``rows``.

    Notes
    -----
    In layout units the right block has width 1. Row ``i`` justifies to it at height
    ``h_i = (1 - (ncol_i - 1) * gap) / sum_j a_ij`` where ``a_ij`` are the projected
    aspects; the block's total height is ``sum_i h_i + gaps``, and the overview
    spans that height at width ``a_main * height``. Those extents are placed as
    figure-fraction rectangles and the figure is sized so one layout unit is the
    same length in x and y, so each rectangle already has its panel's aspect and no
    equal-aspect shrink occurs. Panels are north-anchored, so any residual rounding
    keeps top edges aligned.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from plotea.maps.crs import laea_eu
    >>> fig = plt.figure()
    >>> ax_eu, zooms = panel_mosaic(fig, laea_eu(), main='eu',
    ...     rows=[['fr', 'greater_london'], ['poland_central', 'po_valley']])

    """
    ml, mb, mt, mr = margins
    row_aspects = [[_aspect(b, crs) for b in row] for row in rows]
    row_heights = [(1.0 - (len(a) - 1) * gap) / sum(a) for a in row_aspects]   # justified to unit width
    total_h = sum(row_heights) + (len(rows) - 1) * gap
    main_w = _aspect(main, crs) * total_h
    total_w = main_w + gap + 1.0

    box_w, box_h = 1.0 - ml - mr, 1.0 - mt - mb
    fig.set_size_inches(width, width * (box_w * total_h) / (total_w * box_h))
    sx, sy = box_w / total_w, box_h / total_h

    def rect(x, y, w, h):
        """Layout-unit box (y measured from bottom) -> figure-fraction rectangle."""
        return [ml + x * sx, mb + y * sy, w * sx, h * sy]

    ax_main = carto.new_axes(fig, crs, rect=rect(0.0, 0.0, main_w, total_h))
    ax_main.set_anchor('N')

    zoom_axes, y_top = [], total_h
    for row, aspects, h in zip(rows, row_aspects, row_heights):
        x = main_w + gap
        axrow = []
        for aspect in aspects:
            w = aspect * h
            ax = carto.new_axes(fig, crs, rect=rect(x, y_top - h, w, h))
            ax.set_anchor('N')
            axrow.append(ax)
            x += w + gap
        zoom_axes.append(axrow)
        y_top -= h + gap
    _log.info('main + %d rows (%s zoom panels)', len(rows), sum(len(r) for r in rows))
    return ax_main, zoom_axes
