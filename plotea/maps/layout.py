"""
Panel layouts -- arrange several equal-aspect map axes into aligned rows.

``panel_layout`` lays out a large ``main`` overview beside stacked rows of zoom
panels (the podgorski-and-berg arrangement): every row spans the same width, the
panels within a row share a height, and rows may differ in height. Because a map
axes is locked to equal scaling, its height follows its projected aspect ratio,
not the cell -- so the column widths within a row are made proportional to those
aspects, and each row's height falls out of justifying that row to the common
width. Returns bare ``LonLatAxes``; you draw the basemap and data into them.

``LayoutPreview`` renders candidate arrangements of the same regions as bare
basemaps, so a layout can be chosen before any data is involved.

"""
from pathlib import Path

import matplotlib.pyplot as plt

from plotea.log import get_logger
from plotea.maps import carto
from plotea.maps.base import BaseMap
from plotea.generic.labels import panel_label
from plotea.maps.styles import BASEMAP_GREY
from plotea.maps.crs import laea_eu
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


def panel_layout(fig, crs, main, rows, width: float = 16.0, gap: float = 0.015, margins=(0.05, 0.05, 0.03, 0.01)):
    """
    Lay out a ``main`` overview beside justified ``rows`` of zoom panels; return the axes.

    Every row spans the same width, panels within a row share a height, rows may
    differ in height, and the overview's top edge aligns exactly with the top of
    the upper zoom row. Panels are placed as explicit aspect-matched rectangles
    (not GridSpec cells) so equal-aspect axes cannot drift out of alignment.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to build the layout in. Its size is set to the layout's natural
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
    >>> ax_eu, zooms = panel_layout(fig, laea_eu(), main='eu',
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

# Order matters: candidates deal panels from this list, so it fixes which region is b,
# c, d ... Reads west to east along the top row, then north to south along the bottom.
ZOOMS = ['fr', 'iberia', 'gb', 'fi', 'pl_cz', 'it_north', 'lv_lt']
STYLE = BASEMAP_GREY
BOX   = dict(edgecolor='#e06666', linestyle=(0, (4, 3)), facecolor='none', linewidth=1.2, zorder=5)

class LayoutPreview:
    """
    Draw candidate arrangements of one overview plus several zoom panels.

    Parameters
    ----------
    zooms : list of str
        Region names (keys of plotea's ROIS) for the zoom panels.
    main : str
        Region for the overview panel.
    crs : cartopy.crs.CRS

    Examples
    --------
    >>> LayoutPreview(['fi', 'gb']).save_all(Path('/tmp'))

    """

    def __init__(self, zooms=ZOOMS, main='eu', crs=None, width=16.0):
        self.zooms = list(zooms)
        self.main  = main
        self.crs   = crs or laea_eu()
        self.width = width

    def candidates(self):
        """
        The candidate layouts as ``name -> callable(fig) -> (ax_main, zoom_axes)``.

        Examples
        --------
        >>> sorted(LayoutPreview().candidates())

        """
        return {
            'rows_3_4':   lambda fig: self.rows(fig, self.chunk([3, 4])),
            'rows_4_3':   lambda fig: self.rows(fig, self.chunk([4, 3])),
            'rows_2_2_3': lambda fig: self.rows(fig, self.chunk([2, 2, 3])),
            'rows_3_2_2': lambda fig: self.rows(fig, self.chunk([3, 2, 2])),
            'beside_3_below_4': lambda fig: self.beside_and_below(fig, n_beside=3),
            'grid_4x3':   lambda fig: self.grid(fig, ncols=4, nrows=3, main_span=2),
            'grid_3x3':   lambda fig: self.grid(fig, ncols=3, nrows=3, main_span=2),
        }

    def chunk(self, sizes):
        """
        Deal the zooms into rows of the given sizes, dropping rows left empty.

        The sizes are a wish, not a requirement: with fewer zooms than the pattern asks for
        the last rows come up short or vanish, so a candidate still renders when ``--zooms``
        names three regions instead of seven.

        Examples
        --------
        >>> LayoutPreview(['a', 'b', 'c']).chunk([2, 2, 3])
        [['a', 'b'], ['c']]

        """
        rows, start = [], 0
        for size in sizes:
            row = self.zooms[start:start + size]
            start += size
            if row:
                rows.append(row)
        if start < len(self.zooms):
            rows[-1].extend(self.zooms[start:])       # never silently drop a region
        return rows

    def rows(self, fig, rows):
        """
        True-extent layout: the overview on the left, zooms justified into rows.

        Examples
        --------
        >>> LayoutPreview().rows(plt.figure(), [['fi'], ['gb']])

        """
        ax_main, zoom_axes = panel_layout(fig, self.crs, main=self.main, rows=rows, width=self.width)
        boxes = [b for row in rows for b in row]
        axes  = [ax for row in zoom_axes for ax in row]
        # panel_layout only places the axes; each still needs its own basemap drawn, and
        # without one the axes keeps the projection's global extent (a bare LAEA disc).
        self.basemap(ax_main, self.main)
        for ax, box in zip(axes, boxes):
            self.basemap(ax, box)
        return ax_main, axes, boxes

    def beside_and_below(self, fig, n_beside=3, gap=0.015, margin=0.02):
        """
        Zooms beside the overview at its full height, then a full-width row underneath.

        The panels beside the overview are as tall as it is, so the top block reads as one
        band; the row below is justified to the whole figure width, so its panels come out
        shorter and wider. Two sizes of panel, not seven -- the eye groups them at a glance.

        Parameters
        ----------
        n_beside : int
            How many zooms sit beside the overview; the rest go in the row below.

        Examples
        --------
        >>> LayoutPreview().beside_and_below(plt.figure(), n_beside=3)

        """
        beside, below = self.zooms[:n_beside], self.zooms[n_beside:]
        main_aspect   = _aspect(self.main, self.crs)
        beside_aspect = [_aspect(z, self.crs) for z in beside]
        below_aspect  = [_aspect(z, self.crs) for z in below]

        # Layout units: the overview is 1 tall, so a panel beside it is 1 tall and its own
        # aspect wide. The row below justifies to whatever total width that comes to.
        top_w    = main_aspect + gap + sum(beside_aspect) + (len(beside) - 1) * gap
        below_h  = (top_w - (len(below) - 1) * gap) / sum(below_aspect) if below else 0.0
        total_h  = 1.0 + (gap + below_h if below else 0.0)
        box      = 1.0 - 2 * margin
        fig.set_size_inches(self.width, self.width * (box * total_h) / (top_w * box))
        sx, sy   = box / top_w, box / total_h

        def rect(x, y, w, h):
            """Layout-unit box (y from the bottom) -> figure-fraction rectangle."""
            return [margin + x * sx, margin + y * sy, w * sx, h * sy]

        top_y   = total_h - 1.0
        ax_main = self.axes(fig, rect(0.0, top_y, main_aspect, 1.0), self.main)
        axes, x = [], main_aspect + gap
        for name, aspect in zip(beside, beside_aspect):
            axes.append(self.axes(fig, rect(x, top_y, aspect, 1.0), name))
            x += aspect + gap
        x = 0.0
        for name, aspect in zip(below, below_aspect):
            axes.append(self.axes(fig, rect(x, 0.0, aspect * below_h, below_h), name))
            x += aspect * below_h + gap
        return ax_main, axes, list(beside) + list(below)

    def grid(self, fig, ncols=4, nrows=3, main_span=2, gap=0.012, margin=0.02):
        """
        True grid: every cell the same size, the overview spanning ``main_span`` squared cells.

        The zoom boxes are padded to the overview's aspect so the cells tile exactly; the
        padding only ever adds land around a region, never crops it.

        Examples
        --------
        >>> LayoutPreview().grid(plt.figure(), ncols=4, nrows=3)

        """
        cells  = ncols * nrows - main_span ** 2
        zooms  = self.zooms[:cells]
        if len(self.zooms) > cells:
            _log.warning('Grid %dx%d has room for %d zooms, dropping %d',
                         ncols, nrows, cells, len(self.zooms) - cells)
        aspect = sorted(_aspect(name, self.crs) for name in zooms)[len(zooms) // 2]   # median zoom aspect
        boxes  = [self.pad_to_aspect(name, aspect) for name in zooms]

        # Layout units, where one unit is the same length in x and y (a figure fraction is
        # not: it scales by the figure's width in x and its height in y). A cell is 1 wide
        # and 1/aspect tall; the figure is then sized to the layout's own aspect, so every
        # rectangle already has its panel's aspect and cartopy never shrinks one to fit.
        cell_h  = 1.0 / aspect
        total_w = ncols + (ncols - 1) * gap
        total_h = nrows * cell_h + (nrows - 1) * gap
        box_w, box_h = 1.0 - 2 * margin, 1.0 - 2 * margin
        fig.set_size_inches(self.width, self.width * (box_w * total_h) / (total_w * box_h))
        sx, sy = box_w / total_w, box_h / total_h

        # The overview spans main_span cells AND the gaps between them, so its rectangle is
        # not the cell aspect; pad the region to that or it shrinks and leaves a hole.
        span_aspect = ((main_span + (main_span - 1) * gap) /
                       (main_span * cell_h + (main_span - 1) * gap))
        main_box    = self.pad_to_aspect(self.main, span_aspect)

        def rect(col, row, span=1):
            """Cell (col, row) from the top-left, spanning ``span`` cells -> figure fraction."""
            width  = span + (span - 1) * gap
            height = span * cell_h + (span - 1) * gap
            top    = total_h - row * (cell_h + gap)
            return [margin + col * (1 + gap) * sx, margin + (top - height) * sy,
                    width * sx, height * sy]

        ax_main = self.axes(fig, rect(0, 0, main_span), main_box)
        free    = [(c, r) for r in range(nrows) for c in range(ncols)
                   if not (c < main_span and r < main_span)]
        zoom_axes = [self.axes(fig, rect(c, r), box) for (c, r), box in zip(free, boxes)]
        return ax_main, zoom_axes, boxes

    def pad_to_aspect(self, region, aspect, steps=4):
        """
        Grow a box symmetrically until its projected aspect matches ``aspect``.

        Iterates because the projected aspect is not a fixed multiple of the degree
        ratio -- how much a degree of longitude is worth depends on the latitude.

        Examples
        --------
        >>> LayoutPreview().pad_to_aspect('fi', 0.87)

        """
        xmin, xmax, ymin, ymax = Bbox.from_any(region).extent
        for _ in range(steps):
            current = _aspect([xmin, ymin, xmax, ymax], self.crs)
            if abs(current - aspect) < 0.01:
                break
            if current < aspect:                       # too tall: widen
                grow = (xmax - xmin) * (aspect / current - 1) / 2
                xmin, xmax = xmin - grow, xmax + grow
            else:                                      # too wide: heighten
                grow = (ymax - ymin) * (current / aspect - 1) / 2
                ymin, ymax = ymin - grow, ymax + grow
        return [xmin, ymin, xmax, ymax]

    def axes(self, fig, rect, region):
        """
        Add one map axes at a figure-fraction rectangle and draw its basemap.

        Examples
        --------
        >>> LayoutPreview().axes(plt.figure(), [0, 0, 1, 1], 'eu')

        """
        ax = carto.new_axes(fig, self.crs, rect=rect)   # LonLatAxes: untransformed data is lon/lat
        self.basemap(ax, region)
        return ax

    @staticmethod
    def basemap(ax, region):
        """
        Draw one panel's basemap, which is also what sets its extent.

        Examples
        --------
        >>> LayoutPreview.basemap(ax, 'gb')

        """
        BaseMap(bbox=region, style=STYLE, graticule_labels=False, graticule_step=5).plot(ax=ax)
        return ax

    def draw(self, name):
        """
        Build one candidate: basemaps, zoom outlines on the overview, panel letters.

        Examples
        --------
        >>> fig = LayoutPreview().draw('rows_3_4')

        """
        fig = plt.figure()
        ax_main, zoom_axes, boxes = self.candidates()[name](fig)
        for letter, box in zip('bcdefghij', boxes):
            Bbox.from_any(box).plot(ax=ax_main, **BOX)
            xmin, xmax, ymin, ymax = Bbox.from_any(box).extent
            ax_main.text(xmax, ymax, letter, color=BOX['edgecolor'], fontsize=9, fontweight='bold',
                         va='bottom', ha='right', zorder=6)
        panel_label(ax_main, 'a')
        for letter, ax in zip('bcdefghij', zoom_axes):
            panel_label(ax, letter)
        fig.suptitle(name, x=0.01, ha='left', fontsize=11)
        return fig

    def save_all(self, out_dir):
        """
        Render every candidate to ``out_dir/layout_<name>.png``.

        Examples
        --------
        >>> LayoutPreview().save_all(Path('/tmp/layouts'))

        """
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in self.candidates():
            fig  = self.draw(name)
            path = out_dir / f'layout_{name}.png'
            fig.savefig(path, dpi=110)
            plt.close(fig)
            _log.info('Wrote %s', path)
        return out_dir
