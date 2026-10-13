"""
The ``.plotea`` accessor: what a read carries with it instead of beside it.
"""
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from plotea.maps.raster import Raster

CELL = 0.01

def write(path, values, dtype, nodata=None):
    """
    A raster at a known origin, so an edge coordinate is checkable by hand.

    Examples
    --------
    >>> write(tmp_path / 'x.tif', np.arange(16).reshape(4, 4), 'int16', nodata=-9999)

    """
    profile = dict(driver='GTiff', height=values.shape[0], width=values.shape[1], count=1,
                   dtype=dtype, crs='EPSG:4326', transform=from_origin(9.0, 40.0, CELL, CELL))
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(values.astype(dtype), 1)
    return path

def test_extent_is_the_cell_edges_not_the_centres(tmp_path):
    """
    The whole reason the accessor exists, and the error it stops being silent.

    ``imshow`` places an image by its edges. A DataArray's coordinates are cell centres, so
    a caller computing the extent from ``da.x.min()`` is half a pixel out -- the image draws
    shifted, nothing raises, and the map looks fine.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k cell_edges

    """
    path   = write(tmp_path / 'grid.tif', np.arange(100.0).reshape(10, 10), 'float32')
    window = Raster(path).read_window(every=1)

    assert window.plotea.extent == pytest.approx((9.0, 9.1, 39.9, 40.0))
    # And the trap: the centres are exactly half a cell inside those edges.
    assert float(window.x.min()) == pytest.approx(9.0 + CELL / 2)
    assert float(window.y.max()) == pytest.approx(40.0 - CELL / 2)

def test_bbox_feeds_one_read_into_another(tmp_path):
    """
    The same footprint in the order a region is written, so it can be passed straight back.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k feeds_one_read

    """
    path   = write(tmp_path / 'grid.tif', np.arange(400.0).reshape(20, 20), 'float32')
    raster = Raster(path)
    window = raster.read_window(bbox=[9.05, 39.85, 9.15, 39.95])

    again = raster.read_window(bbox=window.plotea.bbox)
    assert again.plotea.extent == pytest.approx(window.plotea.extent)
    # The two orders differ, which is why both exist.
    west, east, south, north = window.plotea.extent
    assert window.plotea.bbox.extent == pytest.approx((west, east, south, north))

def test_the_file_dtype_survives_the_read(tmp_path):
    """
    An Int16 elevation comes back Int16, with its sentinel declared rather than NaN.

    ``rioxarray``'s ``masked=True`` would promote it to float32 and lose the distinction
    between an integer sentinel and a floating-point NaN, which this package cares about.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k dtype_survives

    """
    values = np.arange(100, dtype='int16').reshape(10, 10)
    values[0, 0] = -9999
    path   = write(tmp_path / 'ints.tif', values, 'int16', nodata=-9999)
    window = Raster(path).read_window(every=1)

    assert window.dtype == np.int16
    assert window.rio.nodata == -9999
    masked = window.plotea.to_masked()
    assert masked.dtype == np.int16
    assert masked.mask[0, 0]
    assert masked.count() == 99

def test_to_masked_catches_a_nan_sentinel_too(tmp_path):
    """
    A comparison cannot find a NaN, because NaN equals nothing including itself.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k nan_sentinel

    """
    values = np.arange(16.0).reshape(4, 4)
    values[1, 1] = np.nan
    path   = write(tmp_path / 'nan.tif', values, 'float32', nodata=float('nan'))
    masked = Raster(path).read_window(every=1).plotea.to_masked()

    assert masked.mask[1, 1]
    assert masked.count() == 15

def test_a_scaled_read_becomes_physical_and_scales_its_nodata(tmp_path):
    """
    Physical units are floats, and the sentinel has to move with them.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k scaled_read

    """
    values = np.full((4, 4), 250, dtype='int16')
    values[0, 0] = -9999
    path   = write(tmp_path / 'cm.tif', values, 'int16', nodata=-9999)
    window = Raster(path, scale=0.01).read_window(every=1)

    assert window.rio.nodata == pytest.approx(-99.99)
    assert float(window[1, 1]) == pytest.approx(2.5)
    assert window.plotea.to_masked().count() == 15

def test_the_window_knows_its_crs(tmp_path):
    """
    Georeferencing travels with the values, which is the point of returning a DataArray.

    Examples
    --------
    >>> pytest tests/test_accessor.py -k knows_its_crs

    """
    path   = write(tmp_path / 'grid.tif', np.ones((5, 5)), 'float32')
    window = Raster(path).read_window(every=1)

    assert isinstance(window, xr.DataArray)
    assert window.rio.crs.to_epsg() == 4326
    assert window.dims == ('y', 'x')
