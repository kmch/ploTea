#!/usr/bin/env python
"""
Render candidate panel layouts as bare basemaps, to choose one before any data is plotted.

Each candidate is written to ``<out_dir>/layout_<name>.png``: the overview with every zoom
box outlined and lettered, and one panel per zoom. Nothing but coastlines, borders and
boxes -- the point is to judge the arrangement, not the content.

Usage
-----
    ./plot_layouts.py OUT_DIR [--zooms fi lv_lt ...] [--main eu] [--width 16]
"""
import argparse
from pathlib import Path

from plotea.maps.layout import ZOOMS, LayoutPreview


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
