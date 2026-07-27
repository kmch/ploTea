"""
Tests for ``Country`` (Natural Earth polygon + ROI bbox) and multi-panel placement.

``Country`` needs the Natural Earth ``admin_0_countries`` shapefile; those tests
read it via cartopy's cache (fetched on first use), so they are not offline-pure
the way the basemap tests are.

"""
import cartopy.crs as ccrs
import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.base import BaseMap
from plotea.maps.registry import ROIS
from plotea.maps.vector import Country


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


def test_country_bbox_is_the_roi():
    """
    ``Country('fr').bbox`` is the mainland ROI box from ``ROIS``, not the polygon's bounds.

    Examples
    --------
    >>> test_country_bbox_is_the_roi()

    """
    fr = Country('fr')
    minx, miny, maxx, maxy = ROIS['fr']
    assert fr.bbox.extent == (minx, maxx, miny, maxy)


def test_country_polygon_clipped_to_roi():
    """
    France resolves via ISO_A2_EH (despite ISO_A2 == '-99') and is clipped to the mainland ROI.

    Examples
    --------
    >>> test_country_polygon_clipped_to_roi()

    """
    fr = Country('fr')
    gdf = fr.data
    assert len(gdf) == 1                      # dissolved to a single feature
    assert gdf.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = gdf.total_bounds
    assert minx > -12 and maxx < 16           # no Guiana (-53) / Reunion (55)
    assert miny > 38 and maxy < 54


def test_country_unknown_code_raises():
    """
    A code with no ``ROIS`` entry fails fast at construction.

    Examples
    --------
    >>> test_country_unknown_code_raises()

    """
    with pytest.raises(KeyError):
        Country('zz')


def test_plot_into_gridspec_cell():
    """
    ``BaseMap.plot(fig=, spec=)`` places each map in its GridSpec cell, side by side.

    Examples
    --------
    >>> test_plot_into_gridspec_cell()

    """
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    _, ax_left = BaseMap(bbox='eu', crs='laea_eu').plot(fig=fig, spec=gs[0, 0])
    _, ax_right = BaseMap(bbox='fr', crs='laea_eu').plot(fig=fig, spec=gs[0, 1])
    assert ax_left.figure is fig and ax_right.figure is fig
    assert type(ax_left).__name__ == 'LonLatAxes'
    assert isinstance(ax_left.projection, ccrs.LambertAzimuthalEqualArea)
    assert ax_left.get_position().x0 < ax_right.get_position().x0
