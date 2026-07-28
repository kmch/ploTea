"""
Tests for the scale bar and inward graticule labels.

"""
import cartopy.crs as ccrs
import matplotlib
import pytest
from cartopy.mpl.gridliner import Gridliner

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plotea.generic.scalebar import _nice, scalebar
from plotea.maps.base import BaseMap
from plotea.maps.carto import new_axes
from plotea.maps.crs import laea_eu


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Close every figure after each test so the Agg backend does not leak.

    Examples
    --------
    >>> # applied automatically to every test in this module

    """
    yield
    plt.close('all')


def test_nice_rounding():
    """
    ``_nice`` snaps to a 1/2/5 x 10^n length.

    Examples
    --------
    >>> test_nice_rounding()

    """
    assert _nice(37) == 50 and _nice(120) == 100 and _nice(8) == 10 and _nice(2.1) == 2


def test_scalebar_drawn_in_projected_metres():
    """
    The bar draws in the axes' metre data coords, not off-map -- the LonLatAxes regression.

    Examples
    --------
    >>> test_scalebar_drawn_in_projected_metres()

    """
    ax = new_axes(plt.figure(), laea_eu())
    ax.set_extent((7.5, 12.6, 44.6, 46.0), crs=ccrs.PlateCarree())
    length = scalebar(ax, location='lower right', segments=4)
    assert length > 0
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(bars) == 4
    # each block renders inside the axes: with the metre coords mis-read as degrees
    # (the LonLatAxes bug) it would project far off the map, outside the axes bbox.
    ax.figure.canvas.draw()
    axbb = ax.get_window_extent()
    for b in bars:
        # unit-square origin -> the block's (x, y) data corner -> display, through the
        # patch's own transform (transData); a PlateCarree mis-stamp lands far outside.
        px, py = b.get_transform().transform((0, 0))
        assert axbb.x0 <= px <= axbb.x1 and axbb.y0 <= py <= axbb.y1


def test_graticule_inward_padding():
    """
    ``graticule_inward`` pulls the gridline labels inside via negative padding.

    Examples
    --------
    >>> test_graticule_inward_padding()

    """
    fig, ax = BaseMap(bbox='fr', crs='laea_eu', graticule_inward=True).plot()
    gl = ax.findobj(Gridliner)[0]
    assert gl.xpadding < 0 and gl.ypadding < 0
