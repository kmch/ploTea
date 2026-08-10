"""
Tests that ``Raster.read`` decimates rather than loading: correctly, and cheaply.

The efficiency claim is the point of the decimated read -- a continental VRT is tens of
gigabytes, and a figure panel wants a few hundred pixels. These check that asking for those
few hundred pixels costs a few hundred pixels' worth of memory, that the values are the
block means rasterio promises for 'average', and that a class raster read with 'nearest'
comes back with its codes intact rather than averaged into nonsense.

"""
import tracemalloc

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from plotea.maps.raster import Raster

SIDE   = 1200                     # source raster side, in pixels
ORIGIN = (10.0, 50.0)             # lon/lat of the top-left corner
PIXEL  = 0.001                    # degrees per pixel
NODATA = -9999.0

@pytest.fixture
def raster_path(tmp_path):
    """
    A synthetic GeoTIFF whose value is its row index, with a nodata block in one corner.

    The row-index ramp makes every aggregate predictable: the mean of a block of rows is
    the mean of their indices, so a decimated read can be checked against arithmetic
    rather than against another implementation of the same thing.

    Examples
    --------
    >>> raster_path(tmp_path)

    """
    data = np.tile(np.arange(SIDE, dtype='float32').reshape(-1, 1), (1, SIDE))
    data[:100, :100] = NODATA
    path = tmp_path / 'ramp.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=SIDE, width=SIDE, count=1,
                       dtype='float32', crs='EPSG:4326', nodata=NODATA,
                       transform=from_origin(ORIGIN[0], ORIGIN[1], PIXEL, PIXEL)) as ds:
        ds.write(data, 1)
    return path


def whole_extent():
    """
    The synthetic raster's full extent, as ``read`` wants it (lon0, lon1, lat0, lat1).

    Examples
    --------
    >>> whole_extent()

    """
    return (ORIGIN[0], ORIGIN[0] + SIDE * PIXEL, ORIGIN[1] - SIDE * PIXEL, ORIGIN[1])


def test_read_decimates_to_max_px(raster_path):
    """
    The longer side comes back at about ``max_px``, whatever the source resolution.

    Examples
    --------
    >>> test_read_decimates_to_max_px(path)

    """
    for max_px in (50, 150, 400):
        data, extent = Raster(raster_path).read(whole_extent(), max_px=max_px)
        assert max(data.shape) == max_px, f'{data.shape} for max_px={max_px}'
        assert extent == whole_extent()          # the extent is echoed unchanged


def test_read_allocates_far_less_than_the_whole_raster(raster_path):
    """
    A decimated read allocates in proportion to what was asked for, not to the file.

    This is the efficiency claim itself: rasterio decimates while reading, so the full
    array never exists. Measured with tracemalloc rather than a clock, so the test does
    not depend on how loaded the machine is.

    Examples
    --------
    >>> test_read_allocates_far_less_than_the_whole_raster(path)

    """
    def peak_bytes(max_px):
        """Peak Python allocation while reading, in bytes."""
        tracemalloc.start()
        Raster(raster_path).read(whole_extent(), max_px=max_px)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    small, full = peak_bytes(100), peak_bytes(SIDE)
    assert small < full / 50, f'decimated read took {small} bytes against {full} for the whole raster'
    assert small < 4 * 100 * 100 * 8            # ~ one 100x100 float array, with slack


def test_read_cost_does_not_follow_the_source_size(tmp_path):
    """
    Doubling the source raster does not double the cost of a fixed-size decimated read.

    Guards the property that matters in practice: the same figure panel costs the same
    whether it is cut from a small file or from a continental mosaic.

    Examples
    --------
    >>> test_read_cost_does_not_follow_the_source_size(tmp_path)

    """
    peaks = {}
    for side in (600, 2400):                     # 16x the pixels
        path = tmp_path / f'ramp_{side}.tif'
        with rasterio.open(path, 'w', driver='GTiff', height=side, width=side, count=1,
                           dtype='float32', crs='EPSG:4326',
                           transform=from_origin(ORIGIN[0], ORIGIN[1], PIXEL, PIXEL)) as ds:
            ds.write(np.zeros((side, side), dtype='float32'), 1)
        extent = (ORIGIN[0], ORIGIN[0] + side * PIXEL, ORIGIN[1] - side * PIXEL, ORIGIN[1])
        tracemalloc.start()
        Raster(path).read(extent, max_px=100)
        peaks[side] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert peaks[2400] < 2 * peaks[600], f'16x the source cost {peaks[2400] / peaks[600]:.1f}x the memory'


def test_average_decimation_gives_block_means(raster_path):
    """
    With ``resampling='average'`` a decimated pixel is the mean of the pixels it covers.

    Examples
    --------
    >>> test_average_decimation_gives_block_means(path)

    """
    factor    = 12
    max_px    = SIDE // factor
    data, _   = Raster(raster_path, resampling='average').read(whole_extent(), max_px=max_px)
    rows      = np.arange(SIDE, dtype=float).reshape(max_px, factor)
    expected  = rows.mean(axis=1)                # the ramp is constant along each row
    assert data.shape == (max_px, max_px)
    np.testing.assert_allclose(data[:, -1], expected, rtol=1e-4)


def test_nearest_decimation_keeps_the_original_values(raster_path):
    """
    With ``resampling='nearest'`` every value read is one that exists in the source.

    That is what a class raster needs: averaging land-cover codes 3 and 5 into 4 invents a
    class that was never there.

    Examples
    --------
    >>> test_nearest_decimation_keeps_the_original_values(path)

    """
    data, _ = Raster(raster_path, resampling='nearest').read(whole_extent(), max_px=100)
    values  = np.unique(data.compressed())
    assert np.all(values == np.round(values))    # row indices are integers; no averaging
    assert set(values.tolist()) <= set(range(SIDE))


def test_nodata_survives_decimation(raster_path):
    """
    The nodata corner comes back masked, not as -9999 smeared into its neighbours.

    Examples
    --------
    >>> test_nodata_survives_decimation(path)

    """
    data, _ = Raster(raster_path, resampling='nearest').read(whole_extent(), max_px=120)
    assert data.mask[:5, :5].all(), 'the nodata corner should be masked'
    assert not data.mask[-5:, -5:].any(), 'the far corner has data'
    assert data.min() >= 0, 'no nodata leaked into the values'


def test_scale_is_applied_after_decimation(raster_path):
    """
    ``scale`` reaches physical units, and does so on the decimated values.

    Examples
    --------
    >>> test_scale_is_applied_after_decimation(path)

    """
    plain, _  = Raster(raster_path).read(whole_extent(), max_px=100)
    scaled, _ = Raster(raster_path, scale=0.01).read(whole_extent(), max_px=100)
    np.testing.assert_allclose(scaled.compressed(), plain.compressed() * 0.01, rtol=1e-6)


def test_window_smaller_than_max_px_is_not_upsampled(raster_path):
    """
    Asking for more pixels than the window holds returns the window, not an interpolation.

    Examples
    --------
    >>> test_window_smaller_than_max_px_is_not_upsampled(path)

    """
    extent  = (ORIGIN[0], ORIGIN[0] + 50 * PIXEL, ORIGIN[1] - 50 * PIXEL, ORIGIN[1])
    data, _ = Raster(raster_path).read(extent, max_px=5000)
    assert data.shape == (50, 50), f'{data.shape}: a 50 px window should stay 50 px'
