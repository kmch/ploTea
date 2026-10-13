"""
Tests for raster drawing: ``hillshaded`` intensity and ``imshow`` on a map axes.

"""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.base import BaseMap
from plotea.maps.raster import RasterPlotter


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


def test_hillshaded_intensity_and_mask():
    """
    ``hillshaded`` returns intensity in [0, 1] and preserves the input mask.

    Examples
    --------
    >>> test_hillshaded_intensity_and_mask()

    """
    dem = _dem()
    shade = RasterPlotter.hillshaded(dem, vert_exag=2.0, dx=90.0, dy=90.0)
    assert shade.shape == dem.shape
    assert 0.0 <= float(shade.min()) and float(shade.max()) <= 1.0
    assert np.array_equal(np.ma.getmaskarray(shade), np.ma.getmaskarray(dem))


def test_imshow_on_lonlat_axes():
    """
    ``imshow`` draws an array over its lon/lat extent with no explicit transform.

    Examples
    --------
    >>> test_imshow_on_lonlat_axes()

    """
    _, ax = BaseMap(bbox='fr', crs='laea_eu').plot()
    shade = RasterPlotter.hillshaded(_dem(), dx=90.0, dy=90.0)
    # no transform= passed: LonLatAxes assumes the extent is lon/lat and it still draws
    im = RasterPlotter.imshow(shade, extent=(-4.762, 9.556, 41.384, 51.097), ax=ax, cmap='gray', vmin=0, vmax=1)
    assert im in ax.images
    ax.figure.canvas.draw()          # would raise if the transform were unresolved


def test_raster_plot_hillshade_switches_the_scale():
    """
    ``Raster.plot(hillshade=True)`` draws 0-1 shading; without it, the values' own range.

    Examples
    --------
    >>> test_raster_plot_hillshade_switches_the_scale()

    """
    from plotea.maps.raster import Raster

    class FakeRaster(Raster):
        """
        A Raster whose data and bounds are synthetic, so the test needs no file.

        ``bounds`` has to be stubbed as well as ``read_window``: ``plot`` asks for
        ``basemap_options`` before it reads any pixels, and that needs the raster's extent,
        which a real Raster gets by opening the file. Stubbing only ``read_window`` left the
        file to be opened for its georeferencing and the test failed on a missing path.

        """

        def read_window(self, bbox=None, max_px=2000, resampling=None):
            return np.ma.array(_dem()), (-4.762, 9.556, 41.384, 51.097)

        @property
        def bounds(self):
            # (xmin, ymin, xmax, ymax), as rioxarray's rio.bounds() returns -- not the
            # (xmin, xmax, ymin, ymax) that read_window hands back for imshow.
            return (-4.762, 41.384, 9.556, 51.097)

    _, ax  = BaseMap(bbox='fr', crs='laea_eu').plot()
    values = FakeRaster('/nonexistent.tif', cmap='terrain', robust=False).plot(ax=ax, vmin=0, vmax=100)
    shaded = FakeRaster('/nonexistent.tif').plot(ax=ax, hillshade=True)
    assert values.get_clim() == (0, 100)
    assert shaded.get_clim() == (0, 1)
    assert shaded.get_array().min() >= 0 and shaded.get_array().max() <= 1


def test_shaded_colour_returns_rgba_with_nodata_transparent():
    """
    Colour blended with relief, and masked cells fully transparent rather than shaded.

    Transparency is the point: a hillshade of nodata is flat grey, which reads as flat
    ground, so an absence would look like a plain.

    Examples
    --------
    >>> pytest tests/test_raster.py -k shaded_colour_returns

    """
    dem  = np.ma.masked_array(np.tile(np.arange(10.0), (10, 1)),
                              mask=np.zeros((10, 10), dtype=bool))
    dem.mask[0, :] = True
    rgb = RasterPlotter.shaded_colour(dem, 'viridis', vert_exag=2, dx=10, dy=10)
    assert rgb.shape == (10, 10, 4)
    assert np.all(rgb[0, :, 3] == 0.0)
    assert np.all(rgb[1:, :, 3] > 0.0)

def test_shaded_colour_refuses_a_discrete_cmap():
    """
    Shading a classified raster moves each class off its own colour, so it is refused.

    Examples
    --------
    >>> pytest tests/test_raster.py -k refuses_a_discrete

    """
    from plotea.maps.cmaps import esa_worldcover

    with pytest.raises(TypeError, match='continuous colormap'):
        RasterPlotter.shaded_colour(np.ones((4, 4)), esa_worldcover())

def test_blend_mode_keeps_the_grey_hillshade_path_intact(tmp_path):
    """
    ``hillshade=True`` alone still draws grey relief; ``blend_mode`` is what changes it.

    Examples
    --------
    >>> pytest tests/test_raster.py -k keeps_the_grey

    """
    import rasterio
    from rasterio.transform import from_origin

    from plotea.maps.raster import Raster

    path = tmp_path / 'ramp.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=40, width=40, count=1,
                       dtype='float32', crs='EPSG:4326',
                       transform=from_origin(9.0, 40.0, 0.01, 0.01)) as dst:
        dst.write(np.tile(np.arange(40.0), (40, 1)).astype('float32'), 1)

    raster  = Raster(path)
    fig, ax = BaseMap(bbox=[9.0, 39.6, 9.4, 40.0]).plot()
    grey    = raster.plot(ax=ax, hillshade=True)
    assert grey.get_array().ndim == 2
    assert grey.get_cmap().name == 'gray'

    blended = raster.plot(ax=ax, hillshade=True, blend_mode='hsv')
    assert blended.get_array().ndim == 3
    assert blended.get_array().shape[2] == 4
