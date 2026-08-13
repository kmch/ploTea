"""
``Colorbar`` -- one colorbar treatment shared by every map, so figures match.

Notes
-----
Two things a bare ``fig.colorbar(mappable, ax=ax)`` gets wrong on a map.

Its height is a guess. A map axes has a fixed aspect ratio -- it must, or the geography
distorts -- so the axes rarely fills the box matplotlib allocated it, and a colorbar sized
from that box overshoots the drawn map, often badly. ``shrink=`` is the usual response and
is a fudge: the right value depends on the data's aspect and changes whenever the view
does.

And its tick labels are formatted per figure, so one panel reads 0.05 and the next
5e-02. Fixing the formatter once here is what makes a set of figures look like a set.

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from plotea.log import get_logger

__all__ = ['Colorbar']

_log = get_logger(__name__)


class Colorbar:
    """
    How a colorbar is attached to a map axes and how its numbers are written.

    Stateless; call :meth:`attach`.

    Examples
    --------
    >>> Colorbar.attach(mesh, ax, label='Radiance (mW m-2 sr-1 nm-1)')

    """

    @staticmethod
    def attach(mappable, ax, label='', width='3%', pad=0.02, labelsize=8, ticksize=7, powerlimits=(-3, 4), orientation='vertical'):
        """
        Add a colorbar whose height matches the drawn map exactly, and return it.

        Parameters
        ----------
        mappable
            What to take the colours from, e.g. the artist returned by ``pcolormesh``.
        ax
            The map axes to sit beside.
        width : str
            Bar thickness as a percentage of the axes width.
        pad : float
            Gap between map and bar, in axes-width fractions.
        powerlimits : tuple of int
            Decade range kept in plain digits. Outside it the ticks switch to a shared
            ``x10^n`` offset rather than each carrying an exponent. See
            :meth:`format_ticks`.

        Returns
        -------
        matplotlib.colorbar.Colorbar

        Notes
        -----
        The bar is an inset axes anchored to ``ax.transAxes``, so it is positioned against
        the axes' own coordinate system rather than against the box matplotlib reserved.
        On a fixed-aspect map those two differ, which is exactly why ``shrink=`` never
        quite lands. Anchoring instead means the bar tracks the map through any later
        change of view or figure size.

        Examples
        --------
        >>> Colorbar.attach(image, ax, label='Pixels per cell')

        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        horizontal = orientation == 'horizontal'
        cax = inset_axes(
            ax,
            width         = '100%' if horizontal else width,
            height        = width if horizontal else '100%',
            loc           = 'lower left',
            bbox_to_anchor= (0.0, -pad - 0.05, 1, 0.05) if horizontal else (1 + pad, 0.0, 1, 1),
            bbox_transform= ax.transAxes,
            borderpad     = 0,
        )
        bar = ax.figure.colorbar(mappable, cax=cax, orientation=orientation)
        bar.set_label(label, fontsize=labelsize)
        bar.ax.tick_params(labelsize=ticksize)
        Colorbar.format_ticks(bar, powerlimits=powerlimits, ticksize=ticksize)
        return bar

    @staticmethod
    def format_ticks(bar, powerlimits=(-3, 4), ticksize=7):
        """
        Write the tick numbers the same way on every figure.

        Plain digits within ``powerlimits`` decades, and beyond them a single ``x10^n``
        carried once at the end of the bar instead of an exponent on every tick.

        Notes
        -----
        Deliberately not scientific notation everywhere. Forcing it turns an honest 25 to
        200 into 2.5x10^1 to 2.0x10^2, which is harder to read and says nothing extra;
        it earns its place only when the numbers are genuinely tiny or huge. Automatic
        switching gives consistency without that cost, and ``useMathText`` renders the
        exponent properly rather than as '1e2'.

        Examples
        --------
        >>> Colorbar.format_ticks(bar, powerlimits=(-3, 4))

        """
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits(powerlimits)
        try:
            bar.ax.yaxis.set_major_formatter(formatter)
            bar.ax.xaxis.set_major_formatter(formatter)
        except AttributeError:  # a categorical or otherwise non-numeric bar
            return bar
        bar.ax.yaxis.get_offset_text().set_fontsize(ticksize)
        bar.ax.xaxis.get_offset_text().set_fontsize(ticksize)
        return bar
