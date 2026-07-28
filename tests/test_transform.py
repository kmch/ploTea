"""
Tests for ``LonLatAxes`` -- untransformed data is assumed to be lon/lat degrees.

The load-bearing property: a plotting call with no ``transform`` must land in the
*same place* as the same call with an explicit ``transform=ccrs.PlateCarree()``,
on any projection, while an explicit ``transform=`` is never overridden and the
cartopy-drawn basemap is left byte-identical.

"""
import inspect

import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as gax
import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import Polygon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps import carto
from plotea.maps.carto import MapAxes, LonLatAxes, draw_basemap, new_axes, _TRANSFORM_METHODS
from plotea.maps.crs import equal_earth, laea_eu

LON = np.array([17.65, -9.14, 24.94])
LAT = np.array([47.79, 38.72, 60.17])


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


def _decorated_methods():
    """
    Return the GeoAxes methods cartopy wraps with ``@_add_transform``, read live from source.

    Examples
    --------
    >>> 'scatter' in _decorated_methods()
    True

    """
    lines = inspect.getsource(gax).splitlines()
    out = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('def ') and i > 0 and lines[i - 1].strip().startswith('@_add_transform'):
            out.add(s.split('(')[0][4:])
    return out


def test_canary_decorated_set_unchanged():
    """
    Pin cartopy's decorated-method set: a change here is silent misplacement.

    Examples
    --------
    >>> test_canary_decorated_set_unchanged()

    """
    assert _decorated_methods() == set(_TRANSFORM_METHODS)


@pytest.mark.parametrize('crs', [equal_earth(), laea_eu(), ccrs.Robinson()])
def test_scatter_bare_matches_explicit(crs):
    """
    Bare ``scatter(lon, lat)`` lands where explicit ``transform=PlateCarree()`` lands, and not where stock GeoAxes puts it.

    Examples
    --------
    >>> test_scatter_bare_matches_explicit(laea_eu())

    """
    ax = new_axes(plt.figure(), crs)
    assert type(ax).__name__ == 'LonLatAxes'
    pts = np.c_[LON, LAT]
    bare = ax.scatter(LON, LAT).get_offset_transform().transform(pts)
    ref = ax.scatter(LON, LAT, transform=ccrs.PlateCarree()).get_offset_transform().transform(pts)
    assert np.abs(bare - ref).max() < 1e-6
    stock = plt.figure().add_subplot(projection=crs)
    wrong = stock.scatter(LON, LAT).get_offset_transform().transform(pts)
    assert np.abs(bare - wrong).max() > 1.0  # the test can actually fail


@pytest.mark.parametrize('crs', [equal_earth(), laea_eu()])
def test_polygon_bare_matches_explicit(crs):
    """
    A geopandas polygon (the ``add_collection`` path, hook 1) lands correctly with no transform.

    Examples
    --------
    >>> test_polygon_bare_matches_explicit(laea_eu())

    """
    poly = gpd.GeoDataFrame(geometry=[Polygon([(2, 48), (10, 48), (10, 54), (2, 54)])], crs=4326)
    ax = new_axes(plt.figure(), crs)
    c_bare = poly.plot(ax=ax).collections[-1]
    c_ref = poly.plot(ax=ax, transform=ccrs.PlateCarree()).collections[-1]
    v_bare = c_bare.get_transform().transform(c_bare.get_paths()[0].vertices)
    v_ref = c_ref.get_transform().transform(c_ref.get_paths()[0].vertices)
    assert np.abs(v_bare - v_ref).max() < 1e-6


def test_explicit_transform_wins():
    """
    An explicit metre-CRS ``transform=`` is honoured, not replaced by the lon/lat default.

    Examples
    --------
    >>> test_explicit_transform_wins()

    """
    ax = new_axes(plt.figure(), laea_eu())
    ax.set_extent([-10, 35, 35, 72], crs=ccrs.PlateCarree())
    x, y = 4321000.0, 3210000.0  # EPSG:3035 metres
    lon, lat = ccrs.PlateCarree().transform_point(x, y, ccrs.epsg(3035))
    sc = ax.scatter([x], [y], transform=ccrs.epsg(3035))
    sc2 = ax.scatter([lon], [lat])  # bare -> lon/lat default
    ax.figure.canvas.draw()
    a = sc.get_offset_transform().transform([[x, y]])
    b = sc2.get_offset_transform().transform([[lon, lat]])
    assert np.abs(a - b).max() < 1.0


def test_data_crs_none_restores_stock():
    """
    Setting ``data_crs = None`` (via a subclass) falls back to stock GeoAxes behaviour.

    Examples
    --------
    >>> test_data_crs_none_restores_stock()

    """
    class _Stock(LonLatAxes):
        data_crs = None

    fig = plt.figure()

    class _P:
        def _as_mpl_axes(self):
            return _Stock, {'projection': laea_eu()}

    ax = fig.add_subplot(1, 1, 1, projection=_P())
    stock = plt.figure().add_subplot(projection=laea_eu())
    pts = np.c_[LON, LAT]
    a = ax.scatter(LON, LAT).get_offset_transform().transform(pts)
    b = stock.scatter(LON, LAT).get_offset_transform().transform(pts)
    assert np.abs(a - b).max() < 1e-6


def test_imshow_lands_projected():
    """
    Bare ``imshow`` on ``LonLatAxes`` warps to projected coords like stock cartopy -- not stamped off-map.

    Regression: hook 1 must skip the ``AxesImage``; otherwise the projected metres
    are read as degrees and the raster lands far off the map (invisible).

    Examples
    --------
    >>> test_imshow_lands_projected()

    """
    arr = np.arange(60 * 80, dtype=float).reshape(60, 80)
    ext = [-4.762, 9.556, 41.384, 51.097]
    ax = new_axes(plt.figure(), laea_eu())
    im = ax.imshow(arr, extent=ext, origin='upper')                    # bare, transform injected
    ref = plt.figure().add_subplot(projection=laea_eu())
    imref = ref.imshow(arr, extent=ext, origin='upper', transform=ccrs.PlateCarree())
    # both warped into projected metres, so the placements coincide and are far from degrees
    assert np.allclose(im.get_extent(), imref.get_extent(), rtol=1e-6)
    assert max(abs(v) for v in im.get_extent()) > 1e5


def test_basemap_pixel_identity():
    """
    The cartopy-drawn basemap is byte-identical on ``LonLatAxes`` and plain ``MapAxes`` -- ``_SKIP`` protects it.

    Examples
    --------
    >>> test_basemap_pixel_identity()

    """
    class _P:
        def __init__(self, cls):
            self.cls = cls

        def _as_mpl_axes(self):
            return self.cls, {'projection': laea_eu()}

    def render(cls):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1, projection=_P(cls))
        draw_basemap(ax, extent=[-10, 35, 35, 72])
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()

    ndiff = int((render(LonLatAxes) != render(MapAxes)).any(axis=-1).sum())
    assert ndiff == 0
