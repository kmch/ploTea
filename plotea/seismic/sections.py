"""
Seismic plots: fields over x and depth, traces over time, single traces and phase plots.

Notes
-----
Depth and time both run down the page, as seismologists read them. Gathers default to
the blue-white-red ``RdBu_r`` scale, white at zero, so positive and negative amplitudes read
apart at a glance, and so do wavefield sections; models are grey. Every image is drawn
with ``imshow`` on the cell edges worked out from the coordinates, so a node sits at the
centre of its pixel rather than on a corner.

"""
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from plotea.log import get_logger
from plotea.maps.colorbar import Colorbar
import xarray as xr

__all__ = ['Gather', 'PhaseMap', 'Section', 'Trace']

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

    def plot(self, ax=None, clip=99.0, vmax=None, cmap='RdBu_r', colorbar=True, label='Amplitude', title=None):
        """
        Draw the gather with a colour scale symmetric about zero and a colorbar fitted to the axes.

        Parameters
        ----------
        clip : float
            Percentile of |amplitude| at which the scale saturates, when ``vmax`` is not given.
        vmax : float, optional
            The scale's limit, -vmax to +vmax; pass the same value to several panels so they
            share one scale.
        colorbar : bool
            Attach a colorbar; leave it off on all but one of a set of panels sharing a scale.

        Examples
        --------
        >>> Gather(gather).plot(clip=98)
        >>> Gather(difference).plot(vmax=1e-8, colorbar=False)

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
        if vmax is None:
            vmax = float(np.percentile(np.abs(da.values), clip)) or 1.0
            _log.info(f'{da.sizes[trace_dim]} traces, clipped at {vmax:.3g} ({clip:g}th percentile of |amplitude|)')
        x0, x1 = _edges(x)
        t0, t1 = _edges(da['time'].values)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        fig = ax.figure
        im  = ax.imshow(da.values, cmap=cmap, vmin=-vmax, vmax=vmax, extent=(x0, x1, t1, t0), aspect='auto', interpolation='nearest')
        ax.set(xlabel=xlabel, ylabel='Time (s)', title=title)
        if colorbar:
            Colorbar.attach(im, ax, label=label)
        return fig, ax


@dataclass(frozen=True, eq=False)
class PhaseMap:
    """
    A phase, in radians, drawn at each receiver's position: a phase plot (Shah et al., 2012).

    Notes
    -----
    ``da`` runs along ``receiver``, with the receivers' positions as coordinates (by default
    ``receiver_x`` and ``receiver_z``, in metres, depth down), as one source's output of
    qcphi's ``PhaseResidual``. Each receiver is one dot, so irregular geometry draws as
    readily as a grid. The scale is ``rainbow`` from -pi to pi: a cycle skip shows as an
    abrupt jump between violet and red, so the receivers where it happens stand out at once.

    Examples
    --------
    >>> fig, ax = PhaseMap(out['residual'].isel(source=0)).plot(source=(500.0, 1000.0))

    """
    da: xr.DataArray

    def __repr__(self):
        return f'PhaseMap(name={self.da.name!r}, receivers={self.da.sizes["receiver"]})'

    def plot(self, ax=None, x='receiver_x', y='receiver_z', depth=True, source=None, outline=None, levels=None, cmap='rainbow', size=None, colorbar=True, label='Phase (rad)', title=None):
        """
        Draw one dot per receiver, coloured by phase, on equal axes in metres, with a colorbar ticked at multiples of pi/2.

        Parameters
        ----------
        x, y : str
            The coordinates holding the receivers' horizontal and vertical positions.
        depth : bool
            ``y`` is depth, so it runs down the page.
        source : tuple of float, optional
            ``(x, y)`` of the source, drawn as a star.
        outline : xarray.DataArray, optional
            A field over (z, x), such as the anomaly, outlined in black at ``levels``.
        size : float, optional
            Dot size in points squared; by default the dots of a regular grid just touch.

        Examples
        --------
        >>> PhaseMap(residual).plot(source=(500.0, 1000.0), outline=anomaly, levels=[-0.1])

        """
        px = self.da[x].values
        py = self.da[y].values
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5))
        fig = ax.figure
        ax.set_aspect('equal')
        ax.set(xlim=_edges(np.unique(px)), ylim=_edges(np.unique(py))[::-1] if depth else _edges(np.unique(py)))
        if size is None:
            step = float(np.diff(np.unique(px)).min()) if np.unique(px).size > 1 else 1.0
            fig.canvas.draw()
            points_per_metre = ax.get_window_extent().width / fig.dpi * 72 / np.ptp(ax.get_xlim())
            size = (step * points_per_metre) ** 2
        sc = ax.scatter(px, py, c=self.da.values, s=size, marker='s', linewidths=0, cmap=cmap, vmin=-np.pi, vmax=np.pi)
        if outline is not None:
            ax.contour(outline['x'].values, outline['z'].values, outline.transpose('z', 'x').values, levels=levels, colors='k', linewidths=0.8)
        if source is not None:
            ax.scatter(*source, marker='*', s=160, c='white', edgecolors='k', linewidths=0.8, zorder=3)
        ax.set(xlabel='X (m)', ylabel='Depth (m)' if depth else 'Y (m)', title=title)
        if colorbar:
            cb = Colorbar.attach(sc, ax, label=label)
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        return fig, ax


@dataclass(frozen=True, eq=False)
class Section:
    """
    A field over x and depth as an image, depth down: a velocity model or a wavefield snapshot.

    Notes
    -----
    ``da`` has dimensions ``z`` and ``x``, in metres, ``z`` positive down. A model reads best
    in a sequential grey scale (``symmetric=False``); a wavefield in a grey scale symmetric
    about zero (``symmetric=True``), white at zero in ``RdBu_r``. The grey scale runs from black
    to white, low to high, so a slow anomaly is dark and black markers stay visible on a fast
    background. A wavefield can be drawn over its model (``background``) to show where it
    meets the structure.

    Examples
    --------
    >>> fig, ax = Section(model.vp).plot(label='P-wave velocity (m/s)')

    """
    da: xr.DataArray

    def __repr__(self):
        return f'Section(name={self.da.name!r}, nz={self.da.sizes["z"]}, nx={self.da.sizes["x"]})'

    def plot(self, ax=None, cmap=None, symmetric=False, clip=99.0, label='', sources=None, receivers=None, title=None, background=None, alpha=None):
        """
        Draw the section with equal axes, a colorbar fitted to it, and optional markers and a field beneath it.

        Parameters
        ----------
        symmetric : bool
            Centre the colour scale on zero and clip it at the ``clip`` percentile of
            |value|, in ``RdBu_r`` by default, for wavefields. Otherwise span the data's range,
            in ``gray`` by default, for models.
        sources, receivers : tuple of array-like, optional
            ``(x, z)`` positions in metres, drawn as stars and triangles.
        background : xarray.DataArray, optional
            A field over the same (z, x), typically the model, drawn beneath in grey so that
            it shows through the section, as in the wavefront-healing animations.
        alpha : float, optional
            Opacity of the section; 0.6 when a background is drawn, 1 otherwise.

        Examples
        --------
        >>> Section(snapshot).plot(symmetric=True, background=model.vp, sources=([1000.], [20.]))

        """
        da = self.da.transpose('z', 'x')
        if symmetric:
            vmax = float(np.percentile(np.abs(da.values), clip)) or 1.0
            vmin = -vmax
            cmap = cmap or 'RdBu_r'
        else:
            vmin = float(da.min())
            vmax = float(da.max())
            cmap = cmap or 'gray'
        x0, x1 = _edges(da['x'].values)
        z0, z1 = _edges(da['z'].values)
        extent = (x0, x1, z1, z0)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        fig = ax.figure
        if background is not None:
            ax.imshow(background.transpose('z', 'x').values, cmap='gray', extent=extent, interpolation='nearest')
            alpha = 0.6 if alpha is None else alpha
        im = ax.imshow(da.values, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest', alpha=alpha)
        if sources is not None:
            ax.scatter(*sources, marker='*', s=90, c='white', edgecolors='k', linewidths=0.8, zorder=3, label='Source')
        if receivers is not None:
            ax.scatter(*receivers, marker='v', s=16, c='white', edgecolors='k', linewidths=0.5, zorder=3, label='Receivers')
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
