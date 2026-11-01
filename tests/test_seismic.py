"""
Seismic plots: sections, gathers and traces from DataArrays, depth and time running down.
"""
import matplotlib
import numpy as np
from plotea.seismic import Gather, Section, Trace
from plotea.seismic.sections import _edges
import pytest
import xarray as xr

matplotlib.use('Agg')

@pytest.fixture
def section():
    """
    A 3 x 4 field over z and x, sampled every 10 m from zero.

    Examples
    --------
    >>> def test_something(section):
    ...     assert section.sizes['x'] == 4

    """
    return xr.DataArray(np.arange(12.0).reshape(3, 4), coords={'z': [0.0, 10.0, 20.0], 'x': [0.0, 10.0, 20.0, 30.0]}, dims=('z', 'x'), name='vp')

def test_edges_lie_half_a_step_beyond_the_nodes():
    """
    A node sits at the centre of its pixel, so the image spans half a step past the first and last nodes.

    Examples
    --------
    >>> pytest tests/test_seismic.py -q

    """
    assert _edges(np.array([0.0, 10.0, 20.0])) == (-5.0, 25.0)

def test_a_section_runs_depth_down_on_the_cell_edges(section):
    """
    The image covers x from -5 to 35 m and depth from 25 m at the bottom to -5 m at the top.

    Examples
    --------
    >>> pytest tests/test_seismic.py -q

    """
    fig, ax = Section(section).plot(label='Value')
    assert ax.images[0].get_extent() == [-5.0, 35.0, 25.0, -5.0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]

def test_a_symmetric_section_centres_the_scale_on_zero(section):
    """
    A wavefield's grey scale runs from -v to +v, so zero is mid-grey whatever the data's range.

    Examples
    --------
    >>> pytest tests/test_seismic.py -q

    """
    fig, ax = Section(section - 3.0).plot(symmetric=True)
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == -vmax

def test_a_gather_puts_traces_at_their_receiver_positions():
    """
    With a ``receiver_x`` coordinate the columns sit at the receivers, and time runs down.

    Examples
    --------
    >>> pytest tests/test_seismic.py -q

    """
    gather = xr.DataArray(np.random.default_rng(0).normal(size=(5, 11)), dims=('receiver', 'time'), coords={'receiver': np.arange(5), 'time': np.arange(11) * 0.1, 'receiver_x': ('receiver', np.arange(5) * 25.0)})
    fig, ax = Gather(gather).plot()
    left, right, bottom, top = ax.images[0].get_extent()
    assert (left, right) == (-12.5, 112.5)
    assert bottom > top
    assert ax.get_xlabel() == 'Receiver x (m)'

def test_a_trace_plots_against_time():
    """
    A wavelet is drawn as one black line over its time axis.

    Examples
    --------
    >>> pytest tests/test_seismic.py -q

    """
    trace = xr.DataArray(np.sin(np.linspace(0, 3, 50)), dims='time', coords={'time': np.linspace(0, 1, 50)}, name='wavelet')
    fig, ax = Trace(trace).plot()
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == 'Time (s)'
