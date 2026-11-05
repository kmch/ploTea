"""
Tests for attaching a colorbar to a map axes.

"""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotea.maps.colorbar import Colorbar


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


def bar_size(bar, ax):
    """
    The bar's width and height as fractions of the map axes' width and height.

    Examples
    --------
    >>> bar_size(Colorbar.attach(image, ax), ax)

    """
    ax.figure.canvas.draw()
    outer, inner = ax.get_position(), bar.ax.get_position()
    return inner.width / outer.width, inner.height / outer.height


def test_vertical_and_horizontal_bars_are_width_thick_beside_and_below_the_map():
    """
    A vertical bar is 3% of the axes wide and spans its height; a horizontal one is 3% of
    the axes high and spans its width, below the map rather than a hairline.

    Examples
    --------
    >>> pytest tests/test_colorbar.py -q

    """
    fig, ax = plt.subplots()
    image   = ax.imshow(np.arange(16.0).reshape(4, 4))

    vertical = Colorbar.attach(image, ax, width='3%')
    assert bar_size(vertical, ax) == pytest.approx((0.03, 1.0), abs=1e-3)
    assert vertical.ax.get_position().x0 > ax.get_position().x1

    horizontal = Colorbar.attach(image, ax, width='3%', pad=0.08, orientation='horizontal')
    assert bar_size(horizontal, ax) == pytest.approx((1.0, 0.03), abs=1e-3)
    assert horizontal.ax.get_position().y1 == pytest.approx(ax.get_position().y0 - 0.08 * ax.get_position().height, abs=1e-3)
