"""
Seismic plots: fields over x and depth, traces over time, and single traces.

Notes
-----
Depth and time both run down the page, as seismologists read them. Every image is drawn
with ``imshow`` on the cell edges worked out from the coordinates, so a node sits at the
centre of its pixel rather than on a corner.

"""
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from plotea.log import get_logger
from plotea.maps.colorbar import Colorbar
import xarray as xr

__all__ = ['Gather', 'Section', 'Trace']

_log = get_logger(__name__)


def _edges(coord):
    """
    The outer edges of a regular coordinate, half a step beyond its first and last values.

    Examples
    --------
    >>> _edges(np.array([0., 10., 20.]))
    (-5.0, 25.0)

    """
    step = float(coord[1] - coord[0]) if coord.size > 1 else 1.0
    return float(coord[0]) - step / 2, float(coord[-1]) + step / 2


@dataclass(frozen=True, eq=False)
class Gather:
    """
    Traces as an image, time down, one column per trace: a shot or receiver gather.

    Notes
    -----
    ``da`` has a trace dimension (``receiver`` for a shot gather) and ``time`` in seconds.
    When the trace dimension carries an ``x`` coordinate (``receiver_x``), the columns sit at
    those positions; otherwise at their index.

    Examples
    --------
    >>> fig, ax = Gather(gather).plot()

    """
    da: xr.DataArray

    def __repr__(self):
        return f'Gather(dims={self.da.dims}, shape={self.da.shape})'

    def plot(self, ax=None, clip=99.0, cmap='gray', title=None):
        """
        Draw the gather with a grey scale symmetric about zero, clipped at a percentile of |amplitude|.

        Examples
        --------
        >>> Gather(gather).plot(clip=98)

        """
        trace_dim = [d for d in self.da.dims if d != 'time'][0]
        x_name    = f'{trace_dim}_x'
        da        = self.da.transpose('time', trace_dim)
        if x_name in da.coords:
            x      = da[x_name].values
            xlabel = f'{trace_dim.capitalize()} x (m)'
        else:
            x      = np.arange(da.sizes[trace_dim], dtype=float)
            xlabel = trace_dim.capitalize()
        vmax = float(np.percentile(np.abs(da.values), clip)) or 1.0
        x0, x1 = _edges(x)
        t0, t1 = _edges(da['time'].values)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        fig = ax.figure
        ax.imshow(da.values, cmap=cmap, vmin=-vmax, vmax=vmax, extent=(x0, x1, t1, t0), aspect='auto', interpolation='nearest')
        ax.set(xlabel=xlabel, ylabel='Time (s)', title=title)
        _log.info(f'{da.sizes[trace_dim]} traces, clipped at {vmax:.3g} ({clip:g}th percentile of |amplitude|)')
        return fig, ax


@dataclass(frozen=True, eq=False)
class Section:
    """
    A field over x and depth as an image, depth down: a velocity model or a wavefield snapshot.

    Notes
    -----
    ``da`` has dimensions ``z`` and ``x``, in metres, ``z`` positive down. A model reads best
    in a sequential grey scale (``symmetric=False``); a wavefield in a grey scale symmetric
    about zero (``symmetric=True``), so that zero is mid-grey. Both run from black to white,
    low to high, so a slow anomaly is dark and black markers stay visible on a fast background.

    Examples
    --------
    >>> fig, ax = Section(model.vp).plot(label='P-wave velocity (m/s)')

    """
    da: xr.DataArray

    def __repr__(self):
        return f'Section(name={self.da.name!r}, nz={self.da.sizes["z"]}, nx={self.da.sizes["x"]})'

    def plot(self, ax=None, cmap=None, symmetric=False, clip=99.0, label='', sources=None, receivers=None, title=None):
        """
        Draw the section with equal axes, a colorbar fitted to it, and optional source and receiver markers.

        Parameters
        ----------
        symmetric : bool
            Centre the colour scale on zero and clip it at the ``clip`` percentile of
            |value|, for wavefields. Otherwise span the data's range, for models.
        sources, receivers : tuple of array-like, optional
            ``(x, z)`` positions in metres, drawn as stars and triangles.

        Examples
        --------
        >>> Section(snapshot).plot(symmetric=True, sources=([1000.], [20.]))

        """
        da = self.da.transpose('z', 'x')
        if symmetric:
            vmax = float(np.percentile(np.abs(da.values), clip)) or 1.0
            vmin = -vmax
            cmap = cmap or 'gray'
        else:
            vmin = float(da.min())
            vmax = float(da.max())
            cmap = cmap or 'gray'
        x0, x1 = _edges(da['x'].values)
        z0, z1 = _edges(da['z'].values)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        fig = ax.figure
        im = ax.imshow(da.values, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x0, x1, z1, z0), interpolation='nearest')
        if sources is not None:
            ax.scatter(*sources, marker='*', s=90, c='white', edgecolors='k', linewidths=0.8, zorder=3, label='Source')
        if receivers is not None:
            ax.scatter(*receivers, marker='v', s=14, c='k', zorder=3, label='Receivers')
        ax.set(xlabel='X (m)', ylabel='Depth (m)', title=title)
        Colorbar.attach(im, ax, label=label)
        return fig, ax


@dataclass(frozen=True, eq=False)
class Trace:
    """
    One trace over time as a line: a source wavelet or a single seismogram.

    Examples
    --------
    >>> fig, ax = Trace(survey.wavelet).plot()

    """
    da: xr.DataArray

    def __repr__(self):
        return f'Trace(name={self.da.name!r}, nt={self.da.sizes["time"]})'

    def plot(self, ax=None, label=None, title=None, **kwargs):
        """
        Draw the trace in black against time in seconds.

        Examples
        --------
        >>> Trace(survey.wavelet).plot(label='Ricker, 10 Hz')

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 2.5))
        fig = ax.figure
        ax.plot(self.da['time'].values, self.da.values, color=kwargs.pop('color', 'k'), lw=kwargs.pop('lw', 1.0), label=label, **kwargs)
        ax.set(xlabel='Time (s)', ylabel='Amplitude', title=title)
        return fig, ax
