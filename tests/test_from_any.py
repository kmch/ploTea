"""
Tests for the uniform ``from_any`` coercion on the map value types.

Every plotea value type absorbs loose input through the same ``from_any`` verb:
``Bbox`` and ``BasemapStyle`` return their own type; ``Crs`` is the one factory
whose return type is foreign (a cartopy CRS), by design.

"""
import cartopy.crs as ccrs
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from plotea.maps.styles import BASEMAP_MUTED, BASEMAP_PLAIN, BasemapStyle
from plotea.maps.crs import Crs, equal_earth
from plotea.maps.vector import Bbox


def test_crs_from_any():
    """
    ``Crs.from_any`` maps None -> Equal Earth, a name -> preset, and passes a live CRS through.

    Examples
    --------
    >>> test_crs_from_any()

    """
    assert isinstance(Crs.from_any(None), ccrs.EqualEarth)
    assert isinstance(Crs.from_any('laea_eu'), ccrs.LambertAzimuthalEqualArea)
    live = ccrs.Robinson()
    assert Crs.from_any(live) is live
    with pytest.raises(KeyError):
        Crs.from_any('nope')
    with pytest.raises(TypeError):
        Crs.from_any(42)


def test_bbox_from_any_always_returns_bbox():
    """
    ``Bbox.from_any`` always returns a ``Bbox``; the whole world is the unbounded state, not None.

    Examples
    --------
    >>> test_bbox_from_any_always_returns_bbox()

    """
    assert Bbox.from_any(None).is_world
    assert Bbox.from_any('world').is_world
    assert Bbox.from_any(None).extent is None
    assert Bbox.from_any('eu').extent == (-10.0, 35.0, 35.0, 72.0)
    assert Bbox.from_any([-10, 35, 35, 72]).extent == (-10.0, 35.0, 35.0, 72.0)
    existing = Bbox([-5, 40, 5, 50], target_crs=4326)
    assert Bbox.from_any(existing) is existing
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs='EPSG:4326')
    assert Bbox.from_any(gdf).extent == pytest.approx((0.0, 10.0, 0.0, 10.0))
    with pytest.raises(KeyError):
        Bbox.from_any('nope')


def test_style_from_any():
    """
    ``BasemapStyle.from_any`` maps None -> plain, a name -> preset, and passes an instance through.

    Examples
    --------
    >>> test_style_from_any()

    """
    assert BasemapStyle.from_any(None) is BASEMAP_PLAIN
    assert BasemapStyle.from_any('muted') is BASEMAP_MUTED
    custom = BasemapStyle(land='black')
    assert BasemapStyle.from_any(custom) is custom
    with pytest.raises(KeyError):
        BasemapStyle.from_any('nope')
    with pytest.raises(TypeError):
        BasemapStyle.from_any(42)
