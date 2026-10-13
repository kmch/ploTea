"""
Tests for the HEALPix layer: cells from points, and the four ways of drawing them.

Skipped entirely without the ``dggs`` extra, which is the one part of plotea with
dependencies of its own (``healpix-geo``, ``healpix-plot``) and the one part a map-only
install has no use for.

"""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

healpix_geo = pytest.importorskip('healpix_geo.nested',
                                  reason="needs the dggs extra: pip install -e '.[dggs]'")

from plotea.maps.dggs.healpix import HealpixCells

LEVEL = 9

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

@pytest.fixture
def points():
    """
    A grid of lon/lat over Sardinia, dense enough that several points share a cell.

    Examples
    --------
    >>> lon, lat = points()

    """
    lon = np.linspace(8.2, 9.8, 120)
    lat = np.linspace(38.9, 41.2, 120)
    lon, lat = np.meshgrid(lon, lat)
    return lon.ravel(), lat.ravel()

@pytest.fixture
def cells(points):
    """
    Cells carrying a value of their own, rather than a point count.

    Examples
    --------
    >>> cells.plot_map()

    """
    lon, lat    = points
    ids         = healpix_geo.lonlat_to_healpix(lon, lat, LEVEL, ellipsoid='WGS84')
    unique      = np.unique(ids)
    clon, clat  = healpix_geo.healpix_to_lonlat(unique, LEVEL, ellipsoid='WGS84')
    return HealpixCells(unique, LEVEL, values=np.hypot(clon - 9.0, clat - 40.0),
                        label='Distance (deg)')

def test_from_lonlat_values_each_cell_by_how_many_points_fell_in_it(points):
    """
    The coverage picture: which cells a swath touches, and how densely.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k values_each_cell

    """
    lon, lat = points
    density  = HealpixCells.from_lonlat(lon, lat, level=LEVEL)
    assert density.cell_ids.size < lon.size                  # points share cells
    assert density.values.sum() == lon.size                  # and every point is counted
    assert density.values.min() >= 1

def test_from_lonlat_refuses_values_of_its_own(points):
    """
    Its values *are* the counts, so a caller's own would have to replace them.

    Without the explicit refusal this surfaced as ``got multiple values for keyword
    argument 'values'``, which names the symptom and not the cause.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k refuses_values

    """
    lon, lat = points
    with pytest.raises(TypeError, match='cannot also take values'):
        HealpixCells.from_lonlat(lon, lat, level=LEVEL, values=np.ones(lon.size))

def test_the_cells_tiling_the_extent_are_mostly_present(cells):
    """
    ``fill_fraction`` is what warns that a resampled image would come out blank.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k mostly_present

    """
    assert 0.0 < cells.fill_fraction <= 1.0

def test_plot_draws_a_collection_of_cells(cells):
    """
    Each cell as a filled quadrilateral, which is the point of the class.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k draws_a_collection

    """
    fig, ax    = plt.subplots()
    collection = cells.plot(ax=ax)
    assert len(collection.get_paths()) == cells.cell_ids.size

def test_plot_map_and_plot_points_both_draw(cells):
    """
    Cells over a basemap, and their centres scattered.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k both_draw

    """
    fig, ax = cells.plot_map()
    assert fig is not None and ax is not None
    assert cells.plot_points() is not None

def test_plot_raster_resamples_and_keeps_the_colorbar_keyword(cells):
    """
    The fine-level route, and that ``cbar`` still reaches healpix-plot.

    This is the regression worth having: healpix-plot spells the keyword ``colorbar`` while
    plotea says ``cbar`` throughout, so the two names meet inside ``plot_raster``. When
    they stopped matching, ``cbar=True`` raised ``unexpected keyword argument`` from a
    third-party call and the failure pointed nowhere near the rename.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k keeps_the_colorbar

    """
    image = cells.plot_raster(shape=128, cbar=True)
    assert image.get_array().size == 128 * 128               # resampled onto the grid asked for
    assert image.axes is not None

def test_plot_raster_refuses_cells_with_no_values(points):
    """
    There is nothing to resample, so say so rather than drawing an empty frame.

    Examples
    --------
    >>> pytest tests/test_dggs.py -k refuses_cells_with_no_values

    """
    lon, lat = points
    ids      = np.unique(healpix_geo.lonlat_to_healpix(lon, lat, LEVEL, ellipsoid='WGS84'))
    with pytest.raises(ValueError, match='needs values'):
        HealpixCells(ids, LEVEL).plot_raster()
