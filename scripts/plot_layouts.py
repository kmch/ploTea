#!/usr/bin/env python
"""
Render candidate panel layouts as bare basemaps, to choose one before any data is plotted.

Each candidate is saved as ``<out_dir>/layout_<name>.png``: the overview with every zoom
box outlined and lettered, and one panel per zoom. Nothing but coastlines, borders and
boxes -- the point is to judge the arrangement, not the content.

Two families are drawn. The ``rows_*`` candidates keep each region's true extent and
justify them into rows (``panel_layout``), so panels share a row height but differ in
width. The ``grid_*`` candidates pad every box to a common aspect first, which gives a
true grid at the cost of showing a little more land around the squarer regions.

Usage
-----
    ./plot_layouts.py OUT_DIR [--zooms fi lv_lt ...] [--width 16]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plotea import BASEMAP_GREY, BaseMap, Bbox, laea_eu, panel_label, panel_layout
from plotea.maps import carto
from plotea.maps.layout import _aspect

ZOOMS = ['fi', 'lv_lt', 'pl_cz', 'it_north', 'gb', 'iberia', 'fr']
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
            print(f'grid {ncols}x{nrows}: {len(self.zooms) - cells} zoom(s) do not fit, dropped')
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
            print('wrote', path)
        return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', type=Path, help='directory the candidate PNGs are written to')
    ap.add_argument('--zooms', nargs='+', default=ZOOMS, help=f'zoom regions (default: {" ".join(ZOOMS)})')
    ap.add_argument('--main', default='eu', help='overview region (default: eu)')
    ap.add_argument('--width', type=float, default=16.0, help='figure width in inches')
    args = ap.parse_args()
    LayoutPreview(zooms=args.zooms, main=args.main, width=args.width).save_all(args.out_dir)


if __name__ == '__main__':
    main()
