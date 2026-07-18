# plotea — final design (maps increment 1)

## 1. Bottom line

The transform policy is an **axes subclass**, not a convention: `LonLatAxes(GeoAxes)` overrides matplotlib's single choke point `_set_artist_props` (catches `add_collection`, `add_patch`, `add_line`, `plot`, `fill` — i.e. all of geopandas' polygon/line paths and the hero-map zoom Rectangle), pre-empts cartopy's `_add_transform` decorator on its **exactly 11** decorated methods (re-verified live: `annotate, barbs, contour, contourf, hexbin, imshow, pcolor, pcolormesh, quiver, scatter, streamplot` — the brief's list is wrong; `plot`, `text`, `fill*`, `tri*` are **not** decorated), and adds a dedicated `text` override because `Axes.text` hardcodes `transform=self.transData` and is structurally invisible to both hooks. It is installed via matplotlib's documented `_as_mpl_axes()` hook, so `Map` never owns the figure — `Map.axes_projection()` drops into GridSpec/`add_axes`/`subplots(subplot_kw=)`/insets, which is what accuml's real layouts need. Untransformed data is assumed lon/lat, guarded by a coordinate range check, logged once per axes at INFO (WARNING when the guard trips), with `transform=` (per artist) and `Map(data_crs=...)` (per axes) as overrides. Everything else is functions + dataclasses: one colorbar (`ax.inset_axes`), one discrete-scale factory, panel labels, save preset, `plot_raster(da, ax=...)`. Two classes total that carry behaviour (`Map`, `LonLatAxes`); the rest is data.

---

## 2. Module layout

```
plotea/
  __init__.py          FILL (0b today). Public surface, one import deep:
                         Map, plot_raster, colorbar, continuous, diverging, class_scale,
                         panel_labels, save, use_style, set_log_level, audit_linewidths,
                         BASEMAP_PLAIN, BASEMAP_MUTED, EXTENTS
  log.py               NEW, top-level. get_logger, set_log_level, _QualNameFilter, _StdoutHandler.
                         Top-level because maps/, generic/, graphs/, tseries/, volumes/ all need it
                         and it depends on nothing.
  style.py             NEW, top-level. use_style, figure_size(columns=), mm(), STYLES_DIR.
  generic/             empty dir today -> gets files. Plot-type-agnostic drawing policy.
    __init__.py        NEW
    colorbar.py        NEW. colorbar() — THE one policy.
    scales.py          NEW. ColorScale, continuous(), diverging(), class_scale().
    labels.py          NEW. panel_labels().
    save.py            NEW. save().
    checks.py          NEW. audit_linewidths(fig), _warn_rainbow(cmap).
  maps/
    __init__.py        NEW (no subpackage has one today).
    base.py            FILL. Map + backend dispatch. NO cartopy types in signatures.
    carto.py           FILL. THE HEART: LonLatAxes, lonlat(), CartopyBackend.draw_basemap().
                         The ONLY file importing cartopy.
    utils.py           FILL. equal_earth(), laea(lon, lat), europe_laea(), EXTENTS, REGIONS,
                         BasemapStyle, BASEMAP_PLAIN, BASEMAP_MUTED, inset().
    raster.py          NEW. plot_raster(da, ax=...).
  graphs/{base,utils}.py     STAY 0-byte. Out of scope: nothing in the accuml call sites this
  tseries/{base,utils}.py    increment targets needs them, and decision 5 keeps ML/stats plots
  volumes/{base,utils}.py    in accuml. They get an __init__.py only, so the package imports.
  legacy/              FROZEN. Untouched. Not re-exported. Not fixed.
styles/
  plotea.mplstyle      EDIT: de-indent + DROP figure.figsize (see §6).
  nature.mplstyle      NEW.
tests/                 NEW.
examples/              NEW.
```

**Packaging reality check**: no subpackage has `__init__.py` and `pyproject.toml` has no `packages` config, so `pip install -e .` today ships **nothing** and `import plotea.maps` cannot work. Step 1 is packaging, not maps.

**Why `base.py` / `carto.py` split**: wsml forked accuml's `plot_map` verbatim and shipped two bugs (the CRS bug at `set_extent(extent, crs=crs)`, *and* `plt` never imported so the `ax=None` branch raises `NameError` — that path has never run). One implementation, one file that knows about cartopy. wsml's `backend=` seam is kept in `Map.__init__` and not built out.

---

## 3. The API

### `plotea/log.py`

```python
import logging
import sys

_FORMAT = '%(levelname)s:%(name)s:%(where)s: %(message)s'


class _QualNameFilter(logging.Filter):
    """
    Inject ``%(where)s`` = 'Class.method' for methods, 'function' for plain functions.

    Notes
    -----
    Attached to the HANDLER, not the logger: filters do not propagate to child
    loggers, so a filter on 'plotea' would miss 'plotea.maps.carto'. On the handler
    it also guarantees ``record.where`` always exists, so the format string cannot
    raise on a third-party record.

    Reads ``co_qualname`` (python >= 3.11) off the caller's code object, matched by
    pathname + funcName. This handles @staticmethod and @classmethod, which
    ``f_locals['self']`` sniffing cannot. Inherited methods report the DEFINING
    class, which is right for locating source.

    Examples
    --------
    >>> set_log_level()
    >>> Map().plot()
    INFO:plotea.maps.base:Map.plot: no transform given; assuming lon/lat (EPSG:4326)

    """

    def filter(self, record: logging.LogRecord) -> bool:
        where = record.funcName
        frame = sys._getframe(0)
        while frame is not None:
            code = frame.f_code
            if code.co_filename == record.pathname and code.co_name == record.funcName:
                where = getattr(code, 'co_qualname', record.funcName)
                break
            frame = frame.f_back
        record.where = where
        return True


class _StdoutHandler(logging.StreamHandler):
    """
    StreamHandler resolving ``sys.stdout`` at emit time, so Jupyter and pytest capture it.

    Notes
    -----
    ``logging.StreamHandler(stream=sys.stdout)`` binds the stream OBJECT once, at
    configure time — which is what accuml/log.py does. Verified: under
    ``redirect_stdout`` the message escapes the redirect entirely. Resolving lazily
    fixes pytest capture and costs nothing.

    Examples
    --------
    >>> h = _StdoutHandler()

    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


def get_logger(name: str) -> logging.Logger:
    """
    Return the module logger. Use as ``_log = get_logger(__name__)`` at module level.

    Examples
    --------
    >>> _log = get_logger(__name__)
    >>> _log.info('assuming lon/lat')

    """
    return logging.getLogger(name)


def set_log_level(level: int = logging.INFO) -> logging.Logger:
    """
    Configure plotea logging: class+method prefixes, visible in Jupyter, no clobbering.

    Notes
    -----
    Attaches to the 'plotea' logger with ``propagate = False``, NOT
    ``basicConfig(force=True)`` on root — force=True rips out the host
    application's handlers, and propagate=False stops Jupyter's own root handler
    printing every message a second time.

    Examples
    --------
    >>> import plotea
    >>> plotea.set_log_level()
    >>> plotea.Map().plot()

    """
```

### `plotea/maps/utils.py`

```python
def equal_earth(central_longitude: float = 0.0) -> 'ccrs.EqualEarth':
    """
    Equal Earth projection — plotea's whole-world default.

    Examples
    --------
    >>> equal_earth()

    """


def laea(lon: float, lat: float) -> 'ccrs.LambertAzimuthalEqualArea':
    """
    Lambert Azimuthal Equal-Area centred on (lon, lat).

    Notes
    -----
    Signature taken from watersmartml's ``crs_lambert(lon, lat)``, which factors
    accuml's zero-argument ``europe_laea()`` correctly.

    Examples
    --------
    >>> laea(10, 52)

    """


def europe_laea() -> 'ccrs.LambertAzimuthalEqualArea':
    """
    LAEA(10, 52) — accuml's workhorse, shipped as a named preset.

    Examples
    --------
    >>> europe_laea()

    """
```

```python
EXTENTS = {
    'world':         None,                    # -> set_global()
    'europe':        (-10, 35, 35, 72),       # accuml plot_europe:76, plot_predictions
    'europe_wide':   (-15, 35, 35, 72),       # accuml plot_predictor_map:94
    'europe_tn':     (-11, 32, 35, 71),       # accuml plot_tn_map:940
    'nitrogen_belt': (-2.0, 9.5, 47.5, 54.0), # hero-map ZOOM
}
# All three near-identical Europe extents are NAMED here. Naming them deletes the
# literals; keeping only one would push two of them back to call sites, which is
# relocation, not deletion.

REGIONS = {
    'world':  dict(projection=equal_earth(), extent=EXTENTS['world']),
    'europe': dict(projection=europe_laea(), extent=EXTENTS['europe']),
}
```

```python
@dataclass(frozen=True)
class BasemapStyle:
    """
    Basemap palette, scale and line weights. Data, not code.

    Notes
    -----
    This is what unifies accuml's two competing basemaps: ONE drawing code path,
    TWO palettes as data. plot_map:18's palette is BASEMAP_PLAIN; basemap:926's is
    BASEMAP_MUTED. Neither is deleted.

    ``scale`` defaults to '50m', NOT cartopy's '110m'. VERIFIED: ne_110m_ocean and
    ne_110m_coastline are absent from ~/.local/share/cartopy, so cfeature defaults
    trigger a network download. This is exactly why accuml's basemap:926
    (``.with_scale("50m")``) is offline-safe and plot_map:18 is not.

    Examples
    --------
    >>> BasemapStyle(land='#efeae1', ocean='#dbe6ec', frame=False)

    """
    land: str = 'white'
    ocean: str = 'whitesmoke'
    coastline: str = '#4d4d4d'
    border: str = '#4d4d4d'
    scale: str = '50m'
    linewidth: float = 0.5
    frame: bool = True


BASEMAP_PLAIN = BasemapStyle()
BASEMAP_MUTED = BasemapStyle(land='#efeae1', ocean='#dbe6ec', coastline='#9fb0b8', border='white', frame=False)
```

```python
def inset(ax, rect: list, extent: tuple, projection=None, mark: bool = True, frame: bool = True):
    """
    Add a zoom inset map, first-class. Returns a LonLatAxes sharing the parent's data_crs.

    Notes
    -----
    Replaces the manual ``fig.add_axes(..., projection=proj)`` + spine loop +
    ``Rectangle`` block in accuml/scripts/plot_hero_tn_map.py. ``mark=True`` draws
    the locator box on the parent — and that box is an ``ax.add_patch(Rectangle)``,
    which only the ``_set_artist_props`` hook can reach.

    Examples
    --------
    >>> fig, ax = Map(region='europe').plot()
    >>> axz = inset(ax, [0.66, 0.03, 0.33, 0.37], extent=EXTENTS['nitrogen_belt'])

    """
```

### `plotea/maps/carto.py`

```python
class LonLatAxes(GeoAxes):
    """
    A cartopy GeoAxes on which UNTRANSFORMED DATA IS LON/LAT (EPSG:4326).

    Cartopy's rule is "no transform= means the data is already in the axes CRS",
    which silently misplaces lon/lat data on every projection except PlateCarree.
    plotea inverts that default, so plain ``gdf.plot(ax=ax)`` is correct on ANY
    projection, and logs once per axes that it made the assumption.

    Notes
    -----
    Three hooks, all verified to 0.00 px on EqualEarth / LAEA(10,52) / Robinson /
    PlateCarree for gdf points, lines, polygons, ax.scatter, ax.plot, ax.add_patch,
    ax.text and da.plot:

    1. ``_set_artist_props`` — matplotlib's single choke point, which stamps
       ``self.transData`` on any artist with an unset transform. Catches
       add_collection / add_patch / add_line / plot / fill.
    2. The 11 methods cartopy decorates with ``_add_transform`` (which defaults to
       ``self.projection``) — pre-set ``kwargs['transform']`` so the default never
       fires. The list is pinned and canaried; the brief's list is wrong.
    3. ``text`` — ``Axes.text`` sets ``transform=self.transData`` EXPLICITLY, so
       ``is_transform_set()`` is True and hook 1 is blind to it. Measured 10-14 px
       silent error on LAEA. Cartopy does not decorate it.

    ``_data_crs`` is a CLASS attribute: ``_set_artist_props`` fires during
    ``GeoAxes.__init__`` (on the title Text artists), before any instance attribute
    would exist.

    Override per artist with ``transform=<crs>``, or per axes with
    ``Map(data_crs=...)``. ``data_crs=None`` restores stock cartopy semantics.

    Examples
    --------
    >>> fig, ax = Map(region='europe').plot()
    >>> gdf.plot(ax=ax)                              # lon/lat assumed, logged once
    >>> gdf_m.plot(ax=ax, transform=laea(10, 52))    # explicit: silent, no assumption

    """
    _data_crs = ccrs.PlateCarree()


def lonlat(crs, data_crs=None):
    """
    Wrap a cartopy CRS so any matplotlib axes factory builds a LonLatAxes.

    Notes
    -----
    Implements matplotlib's documented ``_as_mpl_axes()`` hook, returning
    ``(LonLatAxes, {'projection': crs})`` — the same hook ``ccrs.CRS`` itself uses.
    Return the ``projection`` key, NOT ``map_projection``: the latter is deprecated
    in cartopy 0.25. This is why Map composes with GridSpec, fig.add_axes,
    subplots(subplot_kw=) and insets instead of owning the figure.

    Examples
    --------
    >>> ax = fig.add_subplot(gs[0, 0], projection=lonlat(europe_laea()))

    """
```

### `plotea/maps/base.py`

```python
class Map:
    """
    A basemap specification. ``Map()`` is the whole world with country outlines in Equal Earth.

    Parameters
    ----------
    region : str
        Key into REGIONS: 'world' (default) or 'europe'.
    projection : cartopy CRS, optional
        Overrides the region's projection.
    extent : tuple or str, optional
        (lon_min, lon_max, lat_min, lat_max) in degrees, or a key into EXTENTS.
        None means the region's default; 'world' means global.
    style : BasemapStyle, optional
        Palette / scale / line weights. Defaults to BASEMAP_PLAIN.
    land, ocean, coastline, borders, graticules : bool
        Explicit layer toggles. Named args, not a **kwargs bag. ``borders=True`` is
        what makes "the whole world WITH COUNTRY OUTLINES" a directly assertable
        default rather than a colour string.
    data_crs : cartopy CRS, optional
        CRS assumed for untransformed data. Defaults to lon/lat. None restores
        stock cartopy behaviour.
    backend : str
        Only 'cartopy' is implemented. A seam, not an abstraction.

    Notes
    -----
    Map never owns the figure. ``plot(ax=...)`` draws into any axes and
    ``axes_projection()`` hands the projection spec to any matplotlib factory —
    needed because accuml's plot_predictions is 3 projected maps + 1 PLAIN scatter
    panel in one GridSpec.

    There is deliberately no ``Map.add()``: you chain with ``gdf.plot(ax=ax)``.

    Examples
    --------
    >>> map = Map()
    >>> fig, ax = map.plot()
    >>> gdf.plot(ax=ax)
    >>> plot_raster(da, ax=ax)
    >>> fig, ax = Map(region='europe').plot()
    >>> fig, ax = Map(extent='europe_tn', style=BASEMAP_MUTED).plot()

    """

    def __init__(self, region: str = 'world', projection=None, extent=None, style=None, land: bool = True, ocean: bool = True, coastline: bool = True, borders: bool = True, graticules: bool = True, data_crs=_DEFAULT, backend: str = 'cartopy') -> None:
        ...

    def plot(self, ax=None, figsize=None):
        """
        Draw the basemap and return (fig, ax). Creates a LonLatAxes if ax is None.

        Notes
        -----
        ``figsize=None`` DERIVES a projection-aware default from the drawn extent's
        aspect rather than obeying rcParams' square ``figure.figsize: 6, 6``.
        Measured: EqualEarth global aspect is 2.055, LAEA Europe is 0.964 — no
        single fixed value serves both, and 6,6 wastes ~51% of the frame on the
        mandated world default. Width is capped at a screen-friendly ~10 in.

        Examples
        --------
        >>> fig, ax = Map().plot()
        >>> fig, ax = Map(region='europe').plot(ax=my_geoaxes)

        """

    def axes_projection(self):
        """
        Return the projection spec to pass as ``projection=`` to any axes factory.

        Notes
        -----
        The composition path. Verified against GridSpec.add_subplot, fig.add_axes,
        plt.subplots(subplot_kw=), ax.inset_axes and zoom insets — including the
        mixed case where a plain non-map panel in the same GridSpec stays a normal
        Axes.

        Examples
        --------
        >>> proj = Map(region='europe').axes_projection()
        >>> axes = [fig.add_subplot(gs[i], projection=proj) for i in range(3)]
        >>> for ax in axes: Map(region='europe').plot(ax=ax)

        """
```

### `plotea/maps/raster.py`

```python
def plot_raster(da, ax=None, scale=None, method: str = 'pcolormesh', transform=None, **kwargs):
    """
    Draw a georeferenced xarray DataArray on a map. A function — plotea has no Raster class.

    Parameters
    ----------
    da : xarray.DataArray
        2-D with 1-D coords. x/y dim names are sniffed from
        ('x', 'lon', 'longitude') / ('y', 'lat', 'latitude').
    scale : ColorScale, optional
        Supplies cmap, norm and the colorbar label together.
    method : {'pcolormesh', 'imshow'}
        pcolormesh (default) genuinely reprojects each cell and is correct on any
        projection and on curvilinear/rotated grids. imshow is a faster opt-in path
        for large rectilinear rasters.
    transform : cartopy CRS, optional
        CRS of ``da``. None means lon/lat, consistent with the axes policy.

    Notes
    -----
    VERIFIED: ``da.plot(ax=ax)`` forwards **kwargs to the axes method, so on a
    LonLatAxes it hits the pcolormesh hook and lands at 0.00 px with no transform
    plumbing of its own; on a plain GeoAxes it is off by 233 px (world) to 7.9e6 px
    (LAEA). Without rioxarray a bare DataArray carries NO discoverable CRS
    (attrs {}, encoding {}, no grid_mapping) — so the lon/lat assumption plus the
    range guard is the whole safety net. If rioxarray IS installed, ``da.rio.crs``
    is consulted first, behind a try/except ImportError.

    Examples
    --------
    >>> fig, ax = Map(region='europe').plot()
    >>> im = plot_raster(da, ax=ax, scale=continuous(da.values, robust=True, label='TN (mg L$^{-1}$)'))
    >>> colorbar(im, ax=ax)

    """
```

### `plotea/generic/colorbar.py`

```python
def colorbar(mappable, ax=None, label=None, scale=None, where: str = 'right', size: float = 0.03, pad: float = 0.02, length: float = 0.8, extend: str = 'neither', thousands: bool = True):
    """
    THE colorbar. One policy, in axes coordinates via ``ax.inset_axes``.

    Notes
    -----
    Replaces all six mechanisms accuml grew: fig.add_axes fixed rects (plot.py:133
    and :873), ax.inset_axes (:460), fraction=0.046/pad=0.04 (:333),
    make_axes_locatable (scripts), shrink=0.8 (:672), plt.colorbar(ax=, label=)
    (vector.py:112).

    inset_axes wins for a reason specific to this library: a cartopy GeoAxes has a
    FIXED ASPECT, so the drawn box shrinks inside its subplot slot and only
    axes-relative coordinates follow it. Measured misalignment against the DRAWN
    map box: inset 0.0 px at every figsize; fixed_rect 61.8-133.7 px; fraction_pad
    32.2-236.8 px. This REFUTES accuml's own comment at plot.py:133 — the fixed
    rect does stop the bar shifting with label width, but leaves it misaligned with
    the map by up to 134 px. inset_axes gets both properties, and survives GridSpec
    without figure-level constants.

    ``thousands=True`` applies Nature's 1,000 separator to tick labels.

    Examples
    --------
    >>> sc = ax.scatter(df.longitude, df.latitude, c=df.tn)
    >>> colorbar(sc, ax=ax, label='TN (mg L$^{-1}$)')

    """
```

### `plotea/generic/scales.py`

```python
@dataclass(frozen=True)
class ColorScale:
    """
    A cmap + norm + label, plus how to render its colorbar or legend.

    Notes
    -----
    ``.kw`` is a narrow typed unpack of cmap/norm (and vmin/vmax when continuous)
    straight into any artist call — not a kwargs passthrough bag.

    Examples
    --------
    >>> tn = class_scale(bounds=[0, 1, 2.5, 5, 10, np.inf], colors=TN_COLORS, labels=TN_RANGES)
    >>> sc = ax.scatter(x, y, c=v, **tn.kw)
    >>> tn.legend(ax, title='Total nitrogen', unit='mg L$^{-1}$')

    """
    cmap: object
    norm: object
    label: str = None
    ticks: list = None
    ticklabels: list = None

    @property
    def kw(self) -> dict: ...

    def handles(self) -> list:
        """
        Return real matplotlib legend handles, one swatch per class.

        Notes
        -----
        Replaces accuml's tn_class_legend (plot.py:965), which appends raw
        plt.Rectangle objects to ``fig.patches`` at hand-tuned FIGURE coordinates.
        Real handles move with the layout and survive bbox_inches='tight'.

        Examples
        --------
        >>> ax.legend(handles=tn.handles(), title='TN (mg L$^{-1}$)')

        """

    def legend(self, target, loc='upper right', title=None, unit=None, ncol: int = 1): ...


def continuous(values=None, cmap: str = 'viridis', vmin=None, vmax=None, robust: bool = True, label=None) -> ColorScale:
    """
    A continuous scale. ``robust=True`` clips to the 2nd-98th percentile.

    Notes
    -----
    accuml writes this idiom out verbatim twice (plot.py:435-437 and :726-727) and
    documents a degenerate/mostly-zero-binary fallback in plot_predictor_map's
    docstring without ever implementing it. Implemented here once: falls back to
    min/max when the percentile range collapses.

    Warns via logging when ``cmap`` is a perceptually non-uniform rainbow — Nature
    asks that those be avoided, and a style file cannot see a call-site cmap.

    Examples
    --------
    >>> continuous(np.concatenate([obs, pred]), cmap='YlOrRd', label='TN (mg L$^{-1}$)')

    """


def diverging(values=None, cmap: str = 'RdBu', robust: bool = True, center: float = 0.0, label=None) -> ColorScale:
    """
    A symmetric scale centred on ``center`` — for residuals.

    Notes
    -----
    Replaces the ``res_abs = np.nanpercentile(np.abs(resid), 98); vmin=-res_abs,
    vmax=res_abs`` block, which appears at accuml plot.py:438, plot.py:728 and
    plot_shap.py:122 — three sites, each sitting beside the 2/98 continuous block.
    A plain robust_limits() cannot delete it; this can.

    Examples
    --------
    >>> diverging(obs - pred, label='Residual obs-pred')

    """


def class_scale(colors, values=None, bounds=None, labels=None, nodata=None, label=None) -> ColorScale:
    """
    A discrete class scale. Give it either category ``values`` or interval ``bounds``.

    Parameters
    ----------
    values : sequence, optional
        Categorical codes, e.g. WorldCover's [10, 20, ..., 100]. Bounds are derived
        as midpoints between sorted values.
    bounds : sequence, optional
        Explicit interval edges, e.g. TN's [0, 1, 2.5, 5, 10, inf].
        ``len(bounds) == len(colors) + 1``. Mutually exclusive with ``values``.

    Notes
    -----
    ONE function with an explicit discriminator, not two. accuml hand-rolls this
    twice with genuinely different inputs — worldcover_style:179 passes CATEGORY
    CODES and derives midpoints at :213-221; tn_class_cmap:919 passes explicit
    INTERVAL EDGES to BoundaryNorm. Same concept, two input forms, so it is a
    parameter, not two names.

    plotea ships the MECHANISM. The TN bounds, the WorldCover class values, their
    colours and their display names are domain semantics and STAY in accuml.

    Examples
    --------
    >>> class_scale(values=[10, 20, 30], colors=['#006400', '#FFBB22', '#FFFF4C'], labels=['Tree cover', 'Shrubland', 'Grassland'])
    >>> class_scale(bounds=[0, 1, 2.5, 5, 10, np.inf], colors=TN_COLORS, labels=['<1', '1-2.5', '2.5-5', '5-10', '>10'])

    """
```

### `plotea/generic/labels.py`, `save.py`, `checks.py`

```python
def panel_labels(axes, labels=None, loc: str = 'upper left', offset: tuple = (0.02, 0.98), **kwargs) -> list:
    """
    Label panels a, b, c ... in lower-case bold, per Nature Geoscience.

    Notes
    -----
    Nothing in accuml labels panels anywhere, which is why every multi-panel figure
    there is unpublishable as-is. Font size defaults to ``rcParams['font.size']``
    because the guideline requires the SAME type size as the rest of the figure.
    Placed in AXES coordinates, so the lon/lat transform policy is not involved and
    labels survive bbox_inches='tight'.

    Examples
    --------
    >>> fig, axes = plt.subplots(2, 2)
    >>> panel_labels(axes.ravel())
    >>> panel_labels(axes.ravel(), labels=['a', 'b', 'd'])

    """


def save(fig, path, preset: str = 'print', dpi=None, **kwargs):
    """
    Save a figure, creating parent directories. Presets carry the dpi/bbox/facecolor policy.

    Parameters
    ----------
    preset : {'print', 'screen'}
        'print' -> dpi=300, bbox_inches='tight', facecolor='white'. This freezes the
        idiom accuml converged on by hand, preceded by an open-coded mkdir
        (plot.py:163 and every script). 'screen' -> dpi=150.

    Notes
    -----
    facecolor='white' is Nature's "all display items on a white background" rule.
    ``savefig.bbox`` defaults to None in mpl 3.11, so bbox_inches must stay explicit.

    Examples
    --------
    >>> save(fig, 'figures/hero_tn_europe.png')

    """


def audit_linewidths(fig, minimum: float = 1.0) -> list:
    """
    Log every artist in ``fig`` whose linewidth is below Nature's one-point minimum.

    Notes
    -----
    Exists because rcParams CANNOT save explicit call-site values. accuml hardcodes
    coastline 0.4, borders 0.3, gridlines 0.4, scatter linewidths 0.3, heatmap 0.5,
    hist 0.4, axhline 0.5 and plot_tn_map's edgewidth 0.12 — all invisible to any
    style file. LOGS, never raises: some of those are deliberate (1 pt marker edges
    on thousands of points turn plot_tn_map into a black blob), and this must not be
    automated away.

    Examples
    --------
    >>> use_style('nature')
    >>> fig, ax = Map().plot()
    >>> audit_linewidths(fig)

    """
```

### `plotea/style.py`

```python
def use_style(name: str = 'plotea') -> None:
    """
    Apply a bundled matplotlib style: 'plotea' (screen), 'nature' (publication), or a path.

    Notes
    -----
    Replaces the rcParams dict copy-pasted into accuml/scripts three times
    ({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK}). Logs which
    font actually RESOLVED: Nature asks for Helvetica, which is absent on this box
    (and most Linux boxes), and matplotlib substitutes silently — breaking the "same
    typeface across all figures" rule, which is the exact failure the guideline
    exists to prevent.

    Examples
    --------
    >>> import plotea
    >>> plotea.use_style('nature')

    """


def figure_size(columns: float = 1, aspect: float = 0.75) -> tuple:
    """
    Nature figure width in inches: 1 column = 89 mm = 3.504 in, 2 = 183 mm = 7.205 in.

    Examples
    --------
    >>> figure_size(columns=2, aspect=0.55)
    (7.205, 3.963)

    """
```

---

## 4. The transform mechanism

**Verified in this environment** (matplotlib 3.11.0, cartopy 0.25.0, geopandas 1.1.3).

### What upstream actually does

```python
# cartopy/mpl/geoaxes.py — READ, not guessed
def _add_transform(func):
    def wrapper(self, *args, **kwargs):
        transform = kwargs.get('transform', None)
        if transform is None:
            transform = self.projection        # <- THE default plotea must invert
        kwargs['transform'] = transform
        return func(self, *args, **kwargs)
```

```python
# matplotlib/axes/_base.py::_AxesBase._set_artist_props — READ, re-verified live
def _set_artist_props(self, a):
    a.set_figure(self.get_figure(root=False))
    if not a.is_transform_set():
        a.set_transform(self.transData)        # <- the second default
    a.axes = self
```

Re-verified live in this session:
- `_add_transform` decorates **exactly 11** methods: `annotate, barbs, contour, contourf, hexbin, imshow, pcolor, pcolormesh, quiver, scatter, streamplot`. **The brief's Q2 list is wrong** — it includes `fill, fill_between, text, tricontour, tricontourf, tripcolor` (none decorated) and omits `pcolor`. Do not build the loop from the brief.
- `'text' in GeoAxes.__dict__` → False. `'plot'` → False. `'add_collection'` → False. `'_set_artist_props'` → False. So `plot`, `text`, `add_collection`, `add_patch`, `add_line` are all plain matplotlib — which is exactly why `gdf.plot(ax=ax)` on polygons goes unprotected on stock cartopy.

### The three hooks

```python
_DECORATED = ('annotate', 'barbs', 'contour', 'contourf', 'hexbin', 'imshow',
              'pcolor', 'pcolormesh', 'quiver', 'scatter', 'streamplot')
_SKIP = (FeatureArtist, Gridliner)


class LonLatAxes(GeoAxes):
    _data_crs = ccrs.PlateCarree()      # CLASS attr — fires during GeoAxes.__init__

    # HOOK 1 — matplotlib's choke point. Catches add_collection (gdf lines/polys),
    # add_patch (hero-map zoom Rectangle), add_line, plot, fill.
    def _set_artist_props(self, a):
        if self._data_crs is not None and not a.is_transform_set() and not isinstance(a, _SKIP):
            a.set_transform(self._assume_lonlat(a))
        super()._set_artist_props(a)


# HOOK 2 — pre-empt the decorator, generated in a loop, never hand-written.
for _name in _DECORATED + ('text',):        # HOOK 3 rides the same shape
    _parent = getattr(GeoAxes, _name)
    def _wrapper(self, *a, __parent=_parent, __name=_name, **kw):
        if self._data_crs is not None and kw.get('transform', None) is None:
            kw['transform'] = self._assume_lonlat(None, what=__name + '()')
        return __parent(self, *a, **kw)
    setattr(LonLatAxes, _name, functools.wraps(_parent)(_wrapper))
```

**Why each hook is irreducible:**

| path | reached by | why the others miss it |
|---|---|---|
| gdf polygons/lines (`add_collection`) | Hook 1 | not decorated by cartopy |
| gdf points (`ax.scatter`) | Hook 2 | scatter's geographic coords live in **offsets**, not `paths[0].vertices`; the collection arrives at `add_collection` with `is_transform_set()==True` and an `IdentityTransform` (the MARKER path transform). Hook 1 has nothing to fix. |
| `ax.add_patch(Rectangle)` (hero map) | Hook 1 | a Patch is not a Collection — hooking `add_collection` alone leaves 130 px silent error |
| `ax.plot` | Hook 1 | **not** `_add_transform`-decorated, contrary to the brief |
| `ax.text` | Hook 3 | `Axes.text` sets `transform=self.transData` **explicitly**, so `is_transform_set()` is True and Hook 1 is structurally blind. Not decorated. 10-14 px silent error on LAEA. |
| `da.plot(ax=ax)` | Hook 2 (`pcolormesh`) | free — xarray forwards `**kwargs` to the axes method |

Hooks 1 and 2 are **strictly disjoint**; neither is redundant.

**Why assigning a bare CRS works**: `Artist.get_transform()` and `Collection.get_offset_transform()` both resolve non-`Transform` objects via the `_as_mpl_transform(axes)` duck-typing hook that `ccrs.CRS` implements. Resolve eagerly in Hook 1 (`crs._as_mpl_transform(self)`) — the artist's `.axes` is still None at that moment.

**Why `_SKIP`**: `ax.add_feature` routes cartopy's own `FeatureArtist` through `add_collection` → `_set_artist_props` (FeatureArtist's MRO in 0.25 is `FeatureArtist -> Collection -> ...`). FeatureArtist reprojects internally from `feature.crs`, so stamping PlateCarree on it double-transforms and **silently corrupts the basemap** — measured 8,616 to 34,570 differing pixels vs stock GeoAxes. Guarded: pixel-identical. `Gridliner` is included as cheap insurance and to keep plotea's own graticules out of the log path; note honestly that one judge measured it arriving with `is_transform_set()==True`, i.e. it may be over-guarding. It costs one isinstance.

### Installation

```python
class _LonLatProjection:
    def __init__(self, crs, data_crs=None): self.crs, self.data_crs = crs, data_crs
    def _as_mpl_axes(self): return LonLatAxes, {'projection': self.crs}
```
matplotlib's `_process_projection_requirements` calls `_as_mpl_axes()` on any non-string projection — the same hook `ccrs.CRS` uses (`crs.py:800`). Return `projection`, not `map_projection` (deprecated in 0.25). Do **not** use `register_projection`: it dispatches on a string name and `add_subplot` consumes the `projection` kwarg itself. `add_subplot(projection=..., axes_class=...)` is not an option — matplotlib raises `ValueError: Cannot combine 'axes_class' and 'projection'`.

### Range guard + the log message

```python
def _looks_like_degrees(artist) -> bool:
    # offsets for point artists, path vertices otherwise; all |x|<=180 and |y|<=90
```

One funnel, `_assume_lonlat`, called by all three hooks, logging **once per axes** (an `_assumed` flag) at `stacklevel=` so the prefix reads `LonLatAxes.scatter`, not `LonLatAxes._assume_lonlat`:

```
INFO:plotea.maps.carto:LonLatAxes.add_collection: no transform= given; assuming coordinates
  are lon/lat (EPSG:4326). Pass transform=<ccrs> to override.

WARNING:plotea.maps.carto:LonLatAxes.add_collection: no transform= given and coordinates fall
  outside degree range (x=3086657, y=-2292254), so they are NOT lon/lat. Data is probably in a
  projected CRS: pass transform=<ccrs> or reproject with gdf.to_crs(4326).
```

`logging`, **not** `warnings.warn`, per decision 3.

**Guard honesty** — it is a safety net, never proof:
- catches every realistic metre CRS: 3035 (3.9e6), 3857, UTM 32633, Lambert93 2154, and even a km-rescaled LAEA (y=-765 > 90).
- **defeated** by a CORDEX rotated-pole grid (`+proj=ob_tran`, standard for regional climate model output, squarely in accuml/wsml's domain): bounds `[-172.38, 71.32, 170.39, 83.16]` pass the guard and are wildly misplaced. Also by EPSG:4258/4269 (harmlessly) and a local LAEA centred on its own data.
- the unguarded metre case is largely self-announcing anyway: coords outside PlateCarree's domain transform to **NaN and the geometry silently vanishes**. An empty map is louder than a shifted one. The guard's real value is turning a disappearance into an actionable line.
- `gdf.crs` is **confirmed unavailable** at the artist level: geopandas discards it before constructing the collection (`hasattr(collection, 'crs')` → False, no attribute contains 'crs'). The numbers are all plotea can see.

### Overrides

- **per artist**: `gdf.plot(ax=ax, transform=laea(10, 52))` — verified 0.00 px on data genuinely in LAEA metres, and **silent**. Works by construction: all hooks use `setdefault`/`if is None`.
- **per axes**: `Map(data_crs=None)` → stock cartopy semantics restored.

### Verified error table (reproduced independently by three passes)

| projection | artist | stock GeoAxes | LonLatAxes |
|---|---|---|---|
| EqualEarth | gdf points/lines/polys | 115-201 px | **0.00** |
| LAEA(10,52) | gdf points/lines/polys | 6.3-27.6 px | **0.00** |
| Robinson | gdf points/lines/polys | 116-183 px | **0.00** |
| PlateCarree | all | 0.00 | **0.00** |
| LAEA | `ax.add_patch` | 130.93 px | **0.00** |
| LAEA | `ax.text` | 10.67-14.01 px | **0.00** |
| LAEA | `da.plot` | 7,886,970 px | **0.00** |

The LAEA row is the insidious one: 6-27 px is silent, plausible-looking, and exactly the class of bug the wsml fork shipped.

**Test methodology, load-bearing**: use `get_offset_transform()` for point artists, `get_transform()` otherwise. `Axes.scatter` does `offset_transform=kwargs.pop('transform', self.transData)` then `set_transform(IdentityTransform())` — a naive test asserting on `get_transform()` reports **false ~516 px failures**. Also: never probe (0,0) (a fixed point of EqualEarth → false CORRECT) and never assume vertex order (geopandas reorders polygon rings → phantom 63 px error). Compare *every* source coord against its own projected expectation.

---

## 5. Logging

**Design**: module-level `_log`, a `%(where)s` LogRecord attribute injected by a **handler-level** stack-walking Filter reading `co_qualname`, on a **lazy-stdout** handler attached to the **`plotea` logger** with `propagate=False`.

Format: `%(levelname)s:%(name)s:%(where)s: %(message)s`

```
INFO:plotea.maps.base:plot_raster: raster has no CRS; assuming lon/lat      <- plain function
INFO:plotea.maps.base:Map.plot: no transform given; assuming lon/lat        <- method
INFO:plotea.maps.base:Map.europe: preset -> LAEA(10, 52)                    <- classmethod
INFO:plotea.maps.base:Map.preset: loading preset europe                     <- staticmethod
INFO:plotea.generic.colorbar:colorbar: placing inset colorbar
```

Five decisions, each forced by something measured:

1. **Filter, not LoggerAdapter, not per-class logger.** The requirement is that a plain module-level `_log.info(...)` works unchanged in both functions and methods. An Adapter needs `self._log = LoggerAdapter(...)` in every `__init__`. Only a Filter keeps the call site identical. Cost is paid only after `isEnabledFor` passes, so suppressed DEBUG never walks frames.
2. **`co_qualname`, not `f_locals['self']` sniffing.** py3.11+ (env is 3.13). Gives `'Map.plot'` off the code object. Handles `@staticmethod`, which the `self`/`cls` approach structurally cannot. Documented tradeoff: inherited methods report the **defining** class (`EuropeMap().plot()` → `Map.plot`) — right for locating source.
3. **On the HANDLER, not the logger.** Filters do not propagate to child loggers: a filter on `plotea` would miss `plotea.maps.carto`. On the handler it sees everything and guarantees `record.where` exists, so `%(where)s` cannot raise on a foreign record.
4. **Lazy stream, not `StreamHandler(stream=sys.stdout)`.** Verified: the eager form (accuml/log.py's) binds the stream object once — under `redirect_stdout` the message escaped entirely (`captured: ''`). The lazy property captures correctly under pytest and Jupyter. stdout, not stderr, so notebook output is normal rather than red-boxed.
5. **`propagate=False` on the `plotea` logger, not `basicConfig(force=True)` on root.** force=True rips out the host application's handlers; propagate=False stops Jupyter's own root handler double-printing (verified with a pre-installed root handler). Also `logging.getLogger('matplotlib').setLevel(WARNING)`.

**`stacklevel=` composes with the filter and is mandatory for the transform message.** `_log.info(msg, stacklevel=2)` inside `_assume_lonlat` attributes to `LonLatAxes.scatter`; without it the prefix reads `LonLatAxes._assume_lonlat` — accurate, useless.

This is where plotea deliberately beats accuml: accuml's `%(funcName)s` comes from the code object's `co_name` and knows only `plot`, never `Map`.

---

## 6. Styles

### `styles/plotea.mplstyle` (edit)

**Correction to the brief, verified three times independently: the 4-space indentation is NOT a bug.** matplotlib strips leading whitespace; the file loads cleanly, all 18 keys parse (`figure.figsize -> [6.0, 6.0]`, `font.family -> ['Arial']`), no exception, no warning. De-indent for hygiene. **Do not bill it as a bugfix and do not budget it as one.**

The file's real defect is `figure.figsize: 6, 6`. **DROP the key entirely.** Measured: EqualEarth global aspect 2.055, LAEA Europe extent aspect 0.964 — no fixed value serves both, and 6,6 wastes ~51% of the frame on the mandated world default. GeoAxes enforces the data aspect, so the right value is projection-dependent and a style file cannot know it. `Map.plot(figsize=None)` derives it from the drawn extent, capped at ~10 in wide (10 in × 100 dpi = 1000 px — comfortable in a Jupyter cell).

Second edit: `font.family: Arial` is a single-element list with **no fallback chain**. Replace with the two-key idiom (below).

### `styles/nature.mplstyle` (new)

Guideline → rcParam:

| Nature Geoscience guideline | rcParams |
|---|---|
| sans-serif lettering (e.g. Helvetica) | `font.family: sans-serif` + `font.sans-serif: Helvetica, Arial, Nimbus Sans, Liberation Sans, DejaVu Sans`. **Verified: Helvetica is ABSENT here, Arial IS present** (`/usr/share/fonts/truetype/msttcorefonts/Arial.ttf`). Nimbus Sans and Liberation Sans are metric-compatible clones and near-universal on Linux. Bare `Helvetica` silently degrades to DejaVu Sans — breaking the very rule it encodes. |
| same typeface AND ~same font size across figures | `font.size: 8` fixed in POINTS; size the figure via `figure_size(columns=)`, never rescale text |
| all display items on a WHITE background | `figure.facecolor: white`, `axes.facecolor: white`, `savefig.facecolor: white`, `savefig.transparent: False`. **mpl 3.11 defaults are already correct** (`savefig.facecolor: auto` inherits white), so accuml's `facecolor='white'` is redundant — set them anyway so the style is self-contained against a user's rcParams. |
| avoid excessive boxing | `legend.frameon: False`, `axes.spines.top: False`, `axes.spines.right: False` |
| avoid unnecessary colour / decoration | `axes.grid: False` |
| thinnest lines ≥ 1 point | `lines.linewidth: 1.0`, `axes.linewidth: 1.0`, `grid.linewidth: 1.0`, `patch.linewidth: 1.0`, `xtick.major.width: 1.0`, `xtick.minor.width: 1.0`, `ytick.*` likewise. **mpl 3.11 defaults VIOLATE this**: axes.linewidth 0.8, grid.linewidth 0.8, tick major 0.8, tick minor 0.6. |
| one/two-column format (89 / 183 mm) | **cannot** be a fixed `figure.figsize` — see above. `figure_size(columns=1)` → 3.504 in; `columns=2` → 7.205 in; Nature's 1.5-col ≈ 120 mm = 4.724 in. `savefig.bbox` default is None, so `bbox_inches='tight'` stays an explicit `save()` arg. |

**Guidelines that CANNOT be rcParams and therefore need API support** — this is the list that justifies the library:

| guideline | API |
|---|---|
| panels labelled lower-case **bold** a, b, at the same type size | `panel_labels(axes)`. rcParams has no concept of a panel label. accuml has none anywhere. |
| thinnest line ≥ 1 pt | `audit_linewidths(fig)`. **rcParams cannot save call-site literals**: accuml hardcodes coastline 0.4, borders 0.3, gridlines 0.4, marker edges 0.12. A passive style file is blind to every one. Must LOG, never raise — 1 pt edges on thousands of TN points make a black blob, so those are deliberate. |
| thousands separated by commas (1,000) | `colorbar(thousands=True)` → a tick `Formatter`. Not expressible as an rcParam. |
| avoid rainbows / perceptually non-uniform scales | a denylist checked in `continuous`/`diverging`/`class_scale`, logged at WARNING. `image.cmap: viridis` sets the *default*; it cannot stop `cmap='jet'` at a call site. |
| colour-blind-safe combinations (green/magenta, turquoise/red, yellow/blue) | `axes.prop_cycle` covers line/marker cycling only. Map palettes come from `ColorScale`; the *choice* stays with the caller — plotea warns, it does not police. |
| scale bars, length defined in the LEGEND not on the bar | **deferred, deliberately.** accuml has no scalebars, so there is no call site to design against. Add when a figure needs one. |
| sentence case, SI nomenclature (ms not msec), single space between number and unit | prose conventions. Not automatable. Document; do not fake it. |

`use_style` logs which font actually **resolved**, because silent substitution is exactly the failure the typeface rule exists to prevent.

---

## 7. Packaging

`pyproject.toml` is 7 lines with **no dependencies and no packages config**, so `pip install -e .` currently ships nothing.

```toml
[project]
name = "plotea"
version = "0.1.0"
requires-python = ">=3.11"          # co_qualname is 3.11+; the log filter degrades without it
dependencies = [
  "matplotlib>=3.9,<4",            # upper bound: _set_artist_props is private API
  "numpy",
  "cartopy>=0.23",                 # lower bound: FeatureArtist-is-a-Collection guard
  "geopandas>=1.0",                # artist paths are an implementation detail; pin the floor
  "xarray",
  "shapely",
  "pyproj",
]

[project.optional-dependencies]
raster = ["rioxarray"]             # drags in rasterio/GDAL; NOT a hard dep
dev    = ["pytest", "pytest-cov"]
docs   = ["sphinx"]

[tool.setuptools.packages.find]
include = ["plotea*"]

[tool.setuptools.package-data]
plotea = ["../styles/*.mplstyle"]  # or move styles/ under plotea/styles/ — see note
```

Notes:
- **`styles/` is currently outside the package**, so it will not ship in a wheel. Either move it to `plotea/styles/` (cleanest; `use_style` then resolves via `importlib.resources`) or keep it at the repo root and accept that only editable installs see it. **Recommend moving it** — decide at step 2.
- **rioxarray is an OPTIONAL extra, not a dep.** rasterio and rioxarray are both confirmed absent and rasterio is absent from environment.yml. `plot_raster` takes an in-memory DataArray, so neither is *needed*; the extra exists so `da.rio.crs` can be consulted (behind `try/except ImportError`) when a consumer has installed it. Reading GeoTIFFs stays in the consumer package, per decision 6.
- **environment.yml stays the dev/repro environment** (conda, python=3.13, cartopy + geopandas from conda-forge because the GDAL/PROJ stack is painful on pip). `pyproject.dependencies` is the *distribution* contract. Keep them consistent by hand; do not generate one from the other. `environment.yml` should gain nothing this increment — it already has everything.
- **CI has no cartopy cache** (`~/.local/share/cartopy` is user-level). Offline tests must pin `scale='50m'` *and* mark network tests, or CI downloads everything. The '50m works offline' property is a property of **this machine's populated cache**, not of cartopy — plotea cannot promise offline operation in general, only default to the scale that is cached here and log clearly when a download is triggered (catch `cartopy.io.DownloadWarning`).

---

## 8. Incremental delivery plan

Ten commits. Each is reviewable alone, ends green, and leaves the tree importable. **Steps 1-4 are the spine — review as a block.**

**Step 1 — the thinnest end-to-end contract.** `map = Map(); fig, ax = map.plot(); gdf.plot(ax=ax)` **works after this commit.** Minimal but honest: `pyproject.toml` (packages.find + deps + requires-python), `__init__.py` for `maps/`, `generic/`, `graphs/`, `tseries/`, `volumes/`; `plotea/log.py`; `plotea/maps/carto.py` with `LonLatAxes` (all three hooks, `_SKIP`, range guard, `_assume_lonlat`) + `lonlat()`; `plotea/maps/utils.py` with `equal_earth`, `laea`, `europe_laea`, `EXTENTS`, `REGIONS`, `BasemapStyle`, `BASEMAP_PLAIN`/`BASEMAP_MUTED` (data only, no drawing); `plotea/maps/base.py` with `Map`, `Map.plot`, `axes_projection`, aspect-derived figsize; `CartopyBackend.draw_basemap` in carto.py. No colorbar, no scales, no styles, no raster.
*Rationale for the size*: the mandated chain is not decomposable below this — it needs the axes subclass, a CRS, a basemap and a figure. Everything else is genuinely optional and comes later.

**Step 2 — close the artist coverage + the canaries.** Extend Hook 2 over the pinned 11 + `text`, one pixel test each. `tests/test_mpl_contract.py`: assert `_AxesBase._set_artist_props` exists and its source still contains the `is_transform_set` guard; assert cartopy's `_add_transform` set still equals the pinned 11 names. Split from step 1 on purpose: step 1's hooks are verified, step 2's breadth is where a cartopy/mpl upgrade bites.

**Step 3 — styles.** De-indent `plotea.mplstyle`, drop `figure.figsize`, fix the font chain. Add `nature.mplstyle`. Add `plotea/style.py` (`use_style`, `figure_size`, `mm`) and `plotea/generic/checks.py` (`audit_linewidths`, rainbow denylist). Move `styles/` under `plotea/` if that's the decision.

**Step 4 — save + `__init__` re-exports.** `plotea/generic/save.py`. The public surface exists. **Review gate: stop here.** Everything above is hard to reverse; everything below is a leaf.

**Step 5 — `colorbar()`.** The one policy.

**Step 6 — `ColorScale`, `continuous`, `diverging`, `class_scale`.**

**Step 7 — `plot_raster(da, ax=...)`.** Explicitly scheduled — decision 2 mandates it and a plan that never builds it fails the decision.

**Step 8 — `panel_labels()` + `inset()`.**

**Step 9 — the refusal list.** `README.md` / `CONTRIBUTING.md` ships the explicit **DELIBERATELY REFUSED** list (see §10). Not code, but it is the only durable guard on decision 5's boundary and it costs one commit.

**Step 10 — the adoption proof (`examples/` + `tests/test_callsites.py`).** Reconstruct accuml's real call sites against synthetic data, **without importing accuml**: `plot_europe`, `plot_predictor_map` (:94), `plot_predictions`' 3-map+scatter GridSpec (:401), `plot_tn_map` + `tn_class_legend` (:940, :965), `plot_predictors.py`'s pixel→degree tick hack (:61-81), the hero map (`add_axes` + zoom inset + `add_patch`). Acceptance criterion: **each reconstruction must be shorter than the original.**

**Retire the payoff risk early**: do a throwaway reconstruction of `plot_predictor_map` right after step 5. If it isn't obviously shorter, the design is wrong and it costs nothing to find out then.

Sample target (step 10) — `plot_predictor_map`, 44 lines → 5:
```python
def plot_predictor_map(df, col, ax=None):
    fig, ax = Map(region='europe', extent='europe_wide').plot(ax=ax)
    sc = ax.scatter(df.longitude, df.latitude, c=df[col], s=1, **continuous(df[col]).kw)
    colorbar(sc, ax=ax, label=col)
    return fig, ax
```
Gone: `transform=`, `fig.add_axes([0.84, 0.15, 0.02, 0.60])`, the 2/98 percentile block, the extent literal.

---

## 9. Verification

plotea must prove itself standalone (decision 9 — accuml is not modified).

**Step 1**
- `tests/test_transform.py` — **the crown jewel.** Parametrised over {EqualEarth, LAEA(10,52), Robinson, PlateCarree} × {gdf points, gdf lines, gdf polygons, `ax.add_patch`}: plain `gdf.plot(ax=ax)` display coords must land within 1e-6 px of the explicit `transform=ccrs.PlateCarree()` reference. Also assert the **bare GeoAxes case is non-zero** — otherwise the test cannot fail and proves nothing. Uses `get_offset_transform()` for point artists.
- `test_transform.py::test_override_wins` — data genuinely in EPSG:3035, `transform=epsg(3035)` → 0.00 px, and **no log emitted**.
- `test_transform.py::test_data_crs_none` — stock cartopy behaviour restored.
- `test_basemap_pixel_identity` — render a 50m basemap on `LonLatAxes` and on stock `GeoAxes`, diff the RGBA buffers, assert **0 differing pixels**. Unguarded this reads 8,616-34,570. *This is the single most valuable test in the increment*: it is the only thing standing between the user and a silently corrupted basemap on a cartopy upgrade.
- `test_offline` — monkeypatch `NEShpDownloader.acquire_resource` (or block sockets **after** imports; blocking before breaks ssl) and assert `Map().plot()` never enters it. Pins the 50m default so nobody "helpfully" changes it to 110m.
- `test_log.py` — `Map.plot` / `plot_raster` / `Map.europe` (classmethod) / `Map.stat` (staticmethod) prefixes; lazy-stdout capture under `redirect_stdout`; no double-print with a pre-installed root handler; a foreign record does not crash the `%(where)s` formatter.
- `test_map.py` — `Map()` returns `(fig, LonLatAxes)` at EqualEarth spanning the world with `borders=True`; `Map(region='europe')` gives LAEA(10,52) at (-10,35,35,72); `axes_projection()` resolves in GridSpec / `add_axes` / `subplots(subplot_kw=)`, and a plain panel in the same GridSpec stays a normal `Axes`; the basemap emits **zero** lon/lat log lines and user data emits **exactly one**.
- `test_guard.py` — 4326 polygon → INFO; the same polygon `.to_crs(3035)` → WARNING. Plus an **xfail** pinning the known hole: a CORDEX rotated-pole grid passes the guard.
- **runnable example** `examples/01_hello_map.py` — the mandated chain, four lines, prints the log, saves nothing.

**Step 2** — one pixel test per hooked method; the two contract canaries.

**Step 3** — both styles parse; `use_style('nature')` then assert **every** `*.linewidth` rcParam ≥ 1.0 (the Nature rule made executable); `audit_linewidths` flags a deliberately-0.3pt artist; a rainbow cmap logs WARNING; `figure_size(1) == (3.504, ...)`.

**Step 5** — the inset property directly: the colorbar's y-span matches the **drawn map box** within 1 px across figsizes {(6,6), (11,8), (8,4)}, which is exactly what `fixed_rect` (62-134 px) and `fraction_pad` (32-237 px) fail. Plus: geometry is stable when tick labels go `1` → `100,000` (the property accuml's fixed rect was reaching for), and it works in a 2×2 GridSpec.

**Step 6** — **pin against accuml's actual output**, the only behaviour-preservation proof available given accuml cannot be modified: `class_scale(values=WORLDCOVER_VALUES, ...)` reproduces `worldcover_style`'s derived bounds `[9.5, 15.0, 25.0, 35.0, ...]` exactly; `class_scale(bounds=TN_BOUNDS, ...)` reproduces `tn_class_cmap`'s `(cmap, norm)` exactly; `continuous` handles the degenerate mostly-zero-binary case accuml documents but never implemented; `diverging` reproduces the `res_abs` result.

**Step 7** — a lon/lat DataArray lands 0.00 px on LAEA; `da.plot(ax=ax)` inherits the policy for free; `method='imshow'` honoured; `transform=` honoured.

**Step 8** — labels are lower-case, bold, `fontsize == rcParams['font.size']`; `inset` shares the parent's `data_crs`.

**Step 10** — each reconstruction runs offline, renders non-empty, and is asserted **shorter in lines** than the accuml original (line counts hardcoded in the test as documentation).

---

## 10. Open risks

1. **`_set_artist_props` is private matplotlib API — the single biggest risk.** No stability guarantee; could be renamed in any minor release; the failure mode is **silent misplacement**, not an exception. Mitigations: pin `matplotlib>=3.9,<4`; `test_mpl_contract.py` asserts the method exists and still contains the guard; the pixel tests fail loudly. There is **no public choke point** — verified. Hedge worth taking: **also** override the public `add_collection` as belt-and-braces, so the gdf path (the one decision 2 actually mandates) degrades gracefully if the private method vanishes. `add_collection` needs only the `FeatureArtist` guard, a strictly smaller skip-list.

2. **Cartopy's decorated-method set is version-coupled.** Pinned at 11 in 0.25. A 12th decorated method in a future release = silent misplacement. The canary is the whole defence. Same class of risk: the brief's own report already drifted from reality on this list — treat any inherited fact as unverified.

3. **`text` was found by inspection, not systematic search.** It slips through because `Axes.text` sets `transform` explicitly. Any other method doing the same is invisible to all three hooks. Audit before v1: `bar`, `errorbar`, `arrow`, `fill_between`. (`axhline`/`axvline` are arguably fine — axes-relative by intent.) Mitigation: extend the parametrised test to every artist API plotea documents; `transform=` always works.

4. **plotea's rule is not cartopy's rule.** `LonLatAxes` is a `GeoAxes` that behaves differently from a `GeoAxes`. Pasted cartopy examples silently change meaning — correctly, far more often than not, but the surprise is real, and the log fires once per axes, so in a 50-panel loop it's one easy-to-miss line. Mitigations: `data_crs=None`; the class is named `LonLatAxes` so `type(ax)` in a traceback says what it does; the inversion is the **first line** of `Map`'s docstring, the README and every example. This is the direct cost of decision 3, and decision 2 cannot be honoured any other way. **Accept knowingly; document as a headline, not a footnote.**

5. **The range guard is defeatable and the CORDEX case is realistic.** Rotated-pole grids (`+proj=ob_tran`) are the standard for regional climate model output and pass the guard while being wildly misplaced. Local grids near a false origin and km-unit data can also pass. Ship the xfail test as documentation of the hole. `gdf.crs` is provably unreachable at the artist level, so there is no better information available. Consider a `plotea.maps.utils.check_crs(gdf)` helper for consumers who *do* hold the object.

6. **"unless unit is provided" — the one interpretation risk, and it needs a decision before step 1.** All three candidate designs independently read "unit" as "the data's CRS" and expose it as `transform=` (per artist) + `data_crs=` (per axes). Unanimity suggests it's right, but it is the one thing they could be jointly wrong about. **If you meant a literal `unit='m'|'deg'` keyword, say so now** — it is a rename of one parameter today and an API break once examples exist.

7. **Nature's ≥1 pt rule collides head-on with dense maps.** `plot_tn_map` uses `edgewidth=0.12` on thousands of points *because* 1 pt edges make a black blob. `audit_linewidths` will flag real, deliberate choices. It must LOG, never raise, and `nature.mplstyle` must not silently rewrite call-site values it cannot see. Unresolved sub-question: `BasemapStyle.linewidth` defaults to 0.5 (screen-appropriate at world zoom) but Nature wants 1.0 — so `use_style('nature')` would need to override `BasemapStyle`, coupling `style.py` to `maps/utils.py`. **Flag for a decision at step 3.** Expect friction; do not automate it away.

8. **The offline guarantee is a property of this machine.** `~/.local/share/cartopy` is user-level and happens to hold 50m and 10m. On a fresh box or in CI, **every** scale downloads. `scale='50m'` fixes plotea's defaults; it does not make plotea offline-capable. Consider a `Map` preflight that checks the shapefile exists and logs "will download" rather than blocking a socket.
   *Disclosure carried forward*: earlier probing caused `ne_110m_land.*` to be fetched into that cache (mtime 2026-07-17). No repo file was touched. `ne_110m_ocean`/`coastline` remain absent.

9. **Geopandas internals are an implementation detail.** `gdf.plot(ax=ax)` works because geopandas forwards `**style_kwds` into `PatchCollection`/`ax.scatter`. Not a documented contract. Pin `geopandas>=1.0`; keep the gdf cases in the pixel test.

10. **`lonlat()` is not a `ccrs.CRS`.** Consumer code doing `isinstance(projection, ccrs.CRS)` on `axes_projection()`'s return fails. `ax.projection` is still the real CRS, so this bites only code inspecting the spec before axes creation. Document; proxy `__getattr__` to the CRS if it bites.

11. **`plot_raster` uses pcolormesh, which is slow on big rasters.** imshow is far faster but needs a rectilinear grid. `method='imshow'` is offered but **unverified** — do not claim it until tested. Without rioxarray, a DataArray with dims named `x`/`y` in projected metres is silently misplaced unless the caller passes `transform=`. Dim-name sniffing is a heuristic; do not let it become CRS inference.

12. **Surface size.** ~20 public names. Flat (functions + frozen dataclasses, no ABCs, no factory tower) — fwipy died of depth, not breadth — and every name traces to a real accuml call site. Standing constraint: if a name has no call site, delete it. `Map.grid()` was cut for exactly this reason (`axes_projection()` covers accuml's 3-maps-plus-1-scatter case honestly; `grid()` cannot express it). `class_scale` was collapsed from two factories to one with a discriminator for the same reason.

13. **DELIBERATELY REFUSED** — object now, not after the code exists. SHAP / PDP / obs-vs-pred / correlation heatmaps (accuml :351, :473, :678, :825). TN and WorldCover semantics — bounds, class values, colours, DISPLAY_NAMES (:179, :912-915, :978). contextily basemaps (:233 — a network tile dependency). `Raster` / `Vector` classes. rasterio IO. Scale bars and north arrows (Nature asks for scale bars; accuml has none, so there is no call site to design against — add when a figure needs one). Anything in `plotea/legacy/` — frozen, not resurrected, not re-exported; riverpy and fullwavepy are not fixed. `graphs/`, `tseries/`, `volumes/` stay 0-byte this increment.
    **Ship this list in the repo** (step 9). It is the cheapest durable defence of decision 5's boundary, and `ColorScale`, `colorbar` and `panel_labels` are exactly where accuml's domain semantics will try to leak back in. Step 10's examples are the tripwire: if an example needs a TN constant *from plotea*, the boundary has been crossed.
