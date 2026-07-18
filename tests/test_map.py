"""
Tests for ``plotea.BaseMap`` -- the whole-world basemap and the ``Bbox`` view.

"""
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import Polygon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotea
from plotea.maps.base import BaseMap
from plotea.maps.vector import Bbox


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


def _feature_names(ax):
    """
    Return the Natural Earth feature names drawn on a map axes.

    Examples
    --------
    >>> 'admin_0_boundary_lines_land' in _feature_names(ax)

    """
    from cartopy.mpl.feature_artist import FeatureArtist
    names = []
    for artist in ax.get_children():
        if isinstance(artist, FeatureArtist):
            feature = artist._feature
            names.append(getattr(feature, 'name', type(feature).__name__))
    return names


def test_default_map_is_equal_earth_world():
    """
    ``BaseMap().plot()`` returns a GeoAxes spanning the whole world in Equal Earth.

    Examples
    --------
    >>> test_default_map_is_equal_earth_world()

    """
    fig, ax = BaseMap().plot()
    assert isinstance(ax.projection, ccrs.EqualEarth)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x1 > 17e6 and abs(x0) > 17e6
    assert y1 > 8e6 and abs(y0) > 8e6
    assert BaseMap().bbox is None


def test_default_map_has_country_borders():
    """
    The default map draws country borders, and ``borders=False`` removes them.

    Examples
    --------
    >>> test_default_map_has_country_borders()

    """
    fig, ax = BaseMap().plot()
    assert 'admin_0_boundary_lines_land' in _feature_names(ax)

    fig2, ax2 = BaseMap(borders=False).plot()
    assert 'admin_0_boundary_lines_land' not in _feature_names(ax2)


def test_bbox_by_name():
    """
    ``BaseMap(bbox='europe')`` resolves the name to a ``Bbox`` and crops the view.

    Examples
    --------
    >>> test_bbox_by_name()

    """
    bm = BaseMap(bbox='europe')
    assert isinstance(bm.bbox, Bbox)
    assert bm.bbox.extent == (-10.0, 35.0, 35.0, 72.0)
    fig, ax = bm.plot()
    # Default projection is Equal Earth; the view is a small window, not the globe.
    assert isinstance(ax.projection, ccrs.EqualEarth)
    x0, x1 = ax.get_xlim()
    assert abs(x1 - x0) < 1.5e7


def test_bbox_from_bounds():
    """
    A raw ``[minx, miny, maxx, maxy]`` box builds a ``Bbox``.

    Examples
    --------
    >>> test_bbox_from_bounds()

    """
    bm = BaseMap(bbox=[-10, 35, 35, 72])
    assert bm.bbox.extent == (-10.0, 35.0, 35.0, 72.0)


def test_bbox_object_passthrough():
    """
    ``bbox=`` accepts an existing ``Bbox``.

    Examples
    --------
    >>> test_bbox_object_passthrough()

    """
    box = Bbox([-5, 40, 5, 50], target_crs=4326)
    bm = BaseMap(bbox=box)
    assert bm.bbox is box
    assert bm.bbox.extent == (-5.0, 5.0, 40.0, 50.0)


def test_bbox_from_geometry_with_padding():
    """
    ``Bbox`` derives a padded box from a GeoDataFrame.

    Examples
    --------
    >>> test_bbox_from_geometry_with_padding()

    """
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs='EPSG:4326')
    box = Bbox(gdf, pad=0.1)
    assert box.extent == pytest.approx((-1.0, 11.0, -1.0, 11.0))


def test_crs_override():
    """
    ``crs=`` overrides the default Equal Earth projection.

    Examples
    --------
    >>> test_crs_override()

    """
    from plotea.maps.crs import europe_laea
    fig, ax = BaseMap(bbox='europe', crs=europe_laea()).plot()
    assert isinstance(ax.projection, ccrs.LambertAzimuthalEqualArea)


def test_world_string_is_whole_world():
    """
    ``bbox='world'`` is the whole world, same as the default.

    Examples
    --------
    >>> test_world_string_is_whole_world()

    """
    assert BaseMap(bbox='world').bbox is None


def test_plot_bbox_override():
    """
    ``plot(bbox='europe')`` crops a world map for that draw only.

    Examples
    --------
    >>> test_plot_bbox_override()

    """
    bm = BaseMap()
    fig, ax = bm.plot(bbox='europe')
    x0, x1 = ax.get_xlim()
    assert abs(x1 - x0) < 1.5e7
    # The map's own bbox is untouched by the per-call override.
    assert bm.bbox is None


def test_unknown_bbox_raises():
    """
    An unknown bbox name fails loudly, not silently.

    Examples
    --------
    >>> test_unknown_bbox_raises()

    """
    with pytest.raises(KeyError):
        BaseMap(bbox='narnia')


def test_figsize_is_wide_for_world():
    """
    The derived world figsize is wide (aspect ~2), not square.

    Examples
    --------
    >>> test_figsize_is_wide_for_world()

    """
    w, h = BaseMap()._figsize(None)
    assert w > h
    assert 1.8 < w / h < 2.3


def test_figsize_can_be_overridden():
    """
    ``plot(figsize=...)`` overrides the derived size.

    Examples
    --------
    >>> test_figsize_can_be_overridden()

    """
    fig, ax = BaseMap().plot(figsize=(5, 3))
    assert tuple(fig.get_size_inches()) == (5, 3)


def test_plot_into_existing_axes_via_crs():
    """
    ``axes_crs()`` composes into a GridSpec; a plain panel stays a plain Axes.

    Examples
    --------
    >>> test_plot_into_existing_axes_via_crs()

    """
    from cartopy.mpl.geoaxes import GeoAxes
    crs = BaseMap().axes_crs()
    fig = plt.figure()
    map_ax = fig.add_subplot(1, 2, 1, projection=crs)
    plain_ax = fig.add_subplot(1, 2, 2)
    BaseMap(bbox='europe').plot(ax=map_ax)
    assert isinstance(map_ax, GeoAxes)
    assert not isinstance(plain_ax, GeoAxes)


def test_public_api():
    """
    The mandated names are importable from the top-level package.

    Examples
    --------
    >>> test_public_api()

    """
    assert plotea.BaseMap is BaseMap
    assert plotea.Bbox is Bbox
    assert callable(plotea.set_log_level)
    assert callable(plotea.get_logger)
