"""
Tests for raster drawing: ``hillshade`` intensity and ``plot_raster`` on a map axes.

"""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.base import BaseMap
from plotea.maps.raster import hillshade, plot_raster


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


def _dem():
    """
    A small synthetic DEM with an ocean-nodata strip (masked).

    Examples
    --------
    >>> _dem().shape
    (60, 80)

    """
    yy, xx = np.mgrid[0:60, 0:80]
    dem = np.sin(xx / 8.0) * np.cos(yy / 9.0) * 500 + 400
    return np.ma.array(dem, mask=(xx < 10))


def test_hillshade_intensity_and_mask():
    """
    ``hillshade`` returns intensity in [0, 1] and preserves the input mask.

    Examples
    --------
    >>> test_hillshade_intensity_and_mask()

    """
    dem = _dem()
    shade = hillshade(dem, vert_exag=2.0, dx=90.0, dy=90.0)
    assert shade.shape == dem.shape
    assert 0.0 <= float(shade.min()) and float(shade.max()) <= 1.0
    assert np.array_equal(np.ma.getmaskarray(shade), np.ma.getmaskarray(dem))


def test_plot_raster_on_lonlat_axes():
    """
    ``plot_raster`` draws an array over its lon/lat extent with no explicit transform.

    Examples
    --------
    >>> test_plot_raster_on_lonlat_axes()

    """
    _, ax = BaseMap(bbox='fr', crs='laea_eu').plot()
    shade = hillshade(_dem(), dx=90.0, dy=90.0)
    # no transform= passed: LonLatAxes assumes the extent is lon/lat and it still draws
    im = plot_raster(shade, extent=(-4.762, 9.556, 41.384, 51.097), ax=ax, cmap='gray', vmin=0, vmax=1)
    assert im in ax.images
    ax.figure.canvas.draw()          # would raise if the transform were unresolved
