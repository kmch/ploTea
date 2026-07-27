"""
Tests for the discrete colour scale, projected aspect, and panel mosaic.

"""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.generic.labels import panel_label
from plotea.generic.scales import discrete, discrete_colorbar
from plotea.maps.base import BaseMap
from plotea.maps.carto import projected_aspect
from plotea.maps.crs import laea_eu
from plotea.maps.mosaic import panel_mosaic
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


def test_discrete_bands_and_boundaries():
    """
    ``discrete`` yields ``len(bounds) + 1`` colour bands and a matching BoundaryNorm.

    Examples
    --------
    >>> test_discrete_bands_and_boundaries()

    """
    cmap, norm = discrete()                        # default 2/5/10 -> 4 bands
    assert list(norm.boundaries) == [2.0, 5.0, 10.0]
    # 4 distinct band colours: under, two interior, over
    band_colours = {tuple(cmap.get_under())} | {tuple(cmap(i)) for i in range(cmap.N)} | {tuple(cmap.get_over())}
    assert len(band_colours) == 4
    # each band's value maps to its own colour
    assert int(norm(3.0)) == 0 and int(norm(7.0)) == 1
    assert tuple(cmap.get_under()) != tuple(cmap(0))
    with pytest.raises(ValueError):
        discrete(bounds=[5])                        # need >= 2 boundaries


def test_discrete_from_values():
    """
    With ``values`` and no ``bounds``, the boundaries are the interior quantiles.

    Examples
    --------
    >>> test_discrete_from_values()

    """
    vals = np.arange(1, 101)
    cmap, norm = discrete(values=vals, n=4)
    assert len(norm.boundaries) == 3               # n - 1 boundaries for n bands
    assert cmap.N == 2                             # interior colours = boundaries - 1
    assert norm.boundaries[0] < norm.boundaries[1] < norm.boundaries[2]


def test_projected_aspect_positive():
    """
    Projected aspect is a positive width/height ratio; a wide box exceeds a square-ish one.

    Examples
    --------
    >>> test_projected_aspect_positive()

    """
    a_fr = projected_aspect(Bbox.from_any('fr').extent, laea_eu())
    a_po = projected_aspect(Bbox.from_any('po_valley').extent, laea_eu())
    assert a_fr > 0
    assert a_po > a_fr                              # Po valley is much wider than tall


def test_panel_mosaic_shape_and_placement():
    """
    ``panel_mosaic`` returns axes matching the row shape, main left of the zooms.

    Examples
    --------
    >>> test_panel_mosaic_shape_and_placement()

    """
    fig = plt.figure()
    rows = [['fr', 'greater_london'], ['poland_central', 'po_valley']]
    ax_main, zooms = panel_mosaic(fig, laea_eu(), main='eu', rows=rows)
    assert [len(r) for r in zooms] == [2, 2]
    assert type(ax_main).__name__ == 'LonLatAxes'
    # main is left of every zoom
    main_right = ax_main.get_position().x1
    for row in zooms:
        for ax in row:
            assert ax.get_position().x0 >= main_right - 1e-6


def test_panel_mosaic_top_alignment():
    """
    The overview top edge aligns with the top of every panel in the upper zoom row.

    Examples
    --------
    >>> test_panel_mosaic_top_alignment()

    """
    fig = plt.figure()
    ax_main, zooms = panel_mosaic(fig, laea_eu(), main='eu',
        rows=[['fr', 'greater_london'], ['poland_central', 'po_valley']])
    top = ax_main.get_position().y1
    for ax in zooms[0]:
        assert abs(ax.get_position().y1 - top) < 1e-6
    # rows within a row share a height
    h0 = [round(ax.get_position().height, 6) for ax in zooms[0]]
    assert len(set(h0)) == 1


def test_discrete_colorbar_equal_blocks():
    """
    ``discrete_colorbar`` renders one tick per class boundary between equal blocks.

    Examples
    --------
    >>> test_discrete_colorbar_equal_blocks()

    """
    cmap, norm = discrete()                        # boundaries 2/5/10
    fig, ax = plt.subplots()
    cax = ax.inset_axes([0.9, 0.5, 0.03, 0.4])
    cb = discrete_colorbar(fig, cax, cmap, norm, label='TN')
    assert [t.get_text() for t in cb.ax.get_yticklabels()] == ['2', '5', '10']


def test_panel_label_corner():
    """
    ``panel_label`` places the tag in the requested axes corner.

    Examples
    --------
    >>> test_panel_label_corner()

    """
    fig, ax = plt.subplots()
    t = panel_label(ax, 'a', loc='upper left')
    assert t.get_text() == 'a'
    assert t.get_ha() == 'left' and t.get_va() == 'top'


def test_graticule_labels_toggle():
    """
    ``graticule_labels=False`` draws the map with no gridline tick labels.

    Examples
    --------
    >>> test_graticule_labels_toggle()

    """
    fig = plt.figure()
    _, ax = BaseMap(bbox='fr', crs='laea_eu', graticule_labels=False).plot()
    fig.canvas.draw()
    assert not any(t.get_text() for t in ax.texts)
