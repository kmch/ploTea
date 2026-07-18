"""
Pin the offline default: ``BaseMap().plot()`` at resolution '50m' must not trigger a download.

Notes
-----
This is a guard on ``resolution='50m'`` in ``plotea.maps.carto.draw_basemap``. The
50m layers are cached locally; the 110m ocean and coastline may not be, so
cartopy's own default could download. We monkeypatch the Natural Earth
downloader's ``acquire_resource`` to fail loudly, so any accidental switch to an
uncached resolution turns this test red instead of quietly hitting the network.

"""
import cartopy.io.shapereader as shapereader
import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.base import BaseMap


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Close figures after each test.

    Examples
    --------
    >>> # applied automatically

    """
    yield
    plt.close('all')


def test_world_map_renders_offline(monkeypatch):
    """
    Drawing the default world map must never call the Natural Earth downloader.

    Examples
    --------
    >>> # run under pytest with monkeypatch

    """
    def _boom(self, *args, **kwargs):
        raise AssertionError('cartopy attempted a download; the 50m layers should be cached')

    monkeypatch.setattr(shapereader.NEShpDownloader, 'acquire_resource', _boom)
    fig, ax = BaseMap().plot()
    assert ax is not None
