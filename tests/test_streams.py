"""
Tests for ``Streams.plot`` line-width scaling (used to draw connected river networks).

"""
import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import LineString

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.vector import Streams


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


def _streams():
    """
    Three lines with increasing upstream area.

    Examples
    --------
    >>> len(_streams().data)
    3

    """
    gdf = gpd.GeoDataFrame(
        {'UPLAND_SKM': [1.0, 100.0, 10000.0]},
        geometry=[LineString([(i, 0), (i + 1, 1)]) for i in range(3)],
        crs=4326,
    )
    return Streams(data=gdf)


def test_width_by_scales_line_widths():
    """
    ``width_by`` maps the column (log scale) onto ``width_range`` per feature.

    Examples
    --------
    >>> test_width_by_scales_line_widths()

    """
    fig, ax = plt.subplots()
    _streams().plot(ax=ax, width_by='UPLAND_SKM', width_range=(0.2, 1.5))
    lws = list(ax.collections[-1].get_linewidths())
    assert round(min(lws), 3) == 0.2
    assert round(max(lws), 3) == 1.5
    assert all(np.diff(lws) > 0)                    # wider with more upstream area


def test_plain_plot_uses_uniform_width():
    """
    Without ``width_by`` the streams draw at a single line width.

    Examples
    --------
    >>> test_plain_plot_uses_uniform_width()

    """
    fig, ax = plt.subplots()
    _streams().plot(ax=ax, linewidth=0.7)
    lws = set(round(float(w), 3) for w in ax.collections[-1].get_linewidths())
    assert lws == {0.7}
