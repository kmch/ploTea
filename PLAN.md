# plotea: publication-quality plotting library — increment 1 (maps)

## Context

You want to stop rewriting plotting code across seven packages and delegate it to `plotea`, whose purpose is
Nature-journal-quality figures — the reason those guidelines sit in `notebooks/legacy/plotea.ipynb` cells 5–8.
Exploration reshaped the job three ways:

1. **`accuml/plot.py` (997 lines) is the only real source of patterns.** `watersmartml/wsml/plot/carto.py` is a
   verbatim fork of accuml's basemap — a stalled first attempt at this exact extraction — and it shipped two bugs
   while copying: `carto.py:60,67` pass the projected axes CRS to `set_extent`/`gridlines` instead of
   `PlateCarree`, and `plt` is never imported so the `ax=None` branch at `:54` has never run. h2smart, riverpy and
   fullwavepy work in projected coordinates with almost no cartopy. **accuml-first; wsml is a cautionary tale.**
2. **plotea is greenfield.** Every skeleton file is 0 bytes, no subpackage has an `__init__.py`, and
   `pyproject.toml` has no `packages` config — so `pip install -e .` currently ships **nothing** and
   `import plotea.maps` cannot work. Step 1 is packaging, not maps.
3. **Existing integrations are already broken** (all 5 imports do `from plotea.mpl2d import ...`, moved to
   `plotea.legacy.mpl2d`). You chose a clean break; they stay broken until migrated.

The load-bearing finding, verified in the `plotea` env (matplotlib 3.11.0, cartopy 0.25.0, geopandas 1.1.3):

> **`gdf.plot(ax=ax)` is only correct when the axes projection is PlateCarree.** On Robinson it is ~117 px wrong;
> on LAEA Europe only **6.3 px** wrong at world zoom — silent, plausible, catastrophic at country zoom. Exactly
> the bug the wsml fork shipped. `transform=ccrs.PlateCarree()` fixes every case, because matplotlib's
> `Artist.get_transform()` resolves a bare CRS via the `_as_mpl_transform(axes)` hook.

Since `gdf.plot(ax=ax)` must work with no `map.add()`, **plotea must invert cartopy's default assumption**.

## Decisions (settled with you)

| Question | Decision |
| --- | --- |
| Core API | `map = Map(); fig, ax = map.plot()` — whole world with country outlines |
| Chaining | `gdf.plot(ax=ax)`, `plot_raster(da, ax=ax)` verbatim. `map.add(gdf)` **rejected** |
| Transform policy | Untransformed data is **assumed lon/lat**, reported via **logging** |
| `"unit"` | Means **the data's CRS** → `transform=` (per artist), `data_crs=` (per axes). No `unit=` kwarg |
| Default projection | **Equal Earth**; `projection=` optional |
| Europe | **LAEA (10°E, 52°N)** preset — accuml's workhorse |
| Scope | **Generic drawing only.** SHAP/PDP/obs-vs-pred/correlation stay in accuml |
| Domain semantics | TN bounds, WorldCover classes stay in accuml, passed to plotea as *data* |
| Raster | **No `Raster` class.** `plot_raster(da, ax=...)` only |
| Legacy | **Clean break.** No `plotea.mpl2d` shim; `plotea/legacy/` frozen |
| Figure size | Screen-friendly default; Nature in a separate `styles/nature.mplstyle` |

**Why Equal Earth, not Lambert, for the world:** LAEA is excellent for Europe *because it's centred there*;
azimuthal projections degrade away from their centre, and a LAEA world map is a disc whose antipode smears round
the rim. `Map()` means the whole world → Equal Earth (equal-area). LAEA is the Europe preset.

## Conventions (from `~/.claude/CLAUDE.md`)

Arguments on **one line**. numpy docstrings, `"""` on its own line, blank line above the closing `"""`, **always**
an example. Logging: module-level `_log`, `_log.info('...')` in functions *and* methods, Jupyter-visible,
**class methods log class + method**.

## Design

### Module layout

```
plotea/
  __init__.py     FILL — public surface, one import deep
  log.py          NEW  — get_logger, set_log_level, _QualNameFilter, _StdoutHandler
  style.py        NEW  — use_style, figure_size(columns=), mm
  generic/        empty dir today → colorbar.py, scales.py, labels.py, save.py, checks.py
  maps/
    base.py       FILL — Map, backend seam. No cartopy types in signatures
    carto.py      FILL — THE HEART: LonLatAxes, lonlat(), draw_basemap(). Only file importing cartopy
    registry.py   NEW  — EXTENTS, REGIONS, BasemapStyle, BASEMAP_PLAIN/MUTED. Pure data, no drawing,
                         no cartopy import. Mirrors accuml/registry.py (COVARIATE_GROUPS, META_COLS)
    utils.py      FILL — equal_earth(), laea(lon, lat), europe_laea()
    raster.py     NEW  — plot_raster(da, ax=...)
  graphs/ tseries/ volumes/   stay 0-byte; get __init__.py only so the package imports
  legacy/         FROZEN — untouched, not re-exported, not fixed
styles/plotea.mplstyle (edit) + styles/nature.mplstyle (new) → move under plotea/ so they ship in a wheel
tests/ examples/  NEW
```

### The transform mechanism — three hooks on `LonLatAxes(GeoAxes)`

Verified live; **each hook is irreducible** and hooks 1 and 2 are strictly disjoint.

| Hook | Catches | Why the others miss it |
| --- | --- | --- |
| 1. `_set_artist_props` | `add_collection` (gdf lines/polys), `add_patch` (hero-map Rectangle), `add_line`, `plot`, `fill` | not decorated by cartopy |
| 2. cartopy's **11** decorated methods | gdf points via `scatter`; `da.plot` via `pcolormesh` | scatter's coords live in **offsets**; the collection reaches `add_collection` already transform-set (marker-path `IdentityTransform`) — hook 1 has nothing to fix |
| 3. `text` | `ax.text` | sets `transform=self.transData` **explicitly**, so `is_transform_set()` is True and hook 1 is structurally blind |

The 11 decorated methods (pinned, re-verified — **`plot`, `text`, `fill*`, `tri*` are NOT among them**):
`annotate, barbs, contour, contourf, hexbin, imshow, pcolor, pcolormesh, quiver, scatter, streamplot`.

**`_SKIP = (FeatureArtist, Gridliner)` is mandatory.** `ax.add_feature` routes cartopy's own `FeatureArtist`
through `add_collection`; it reprojects internally from `feature.crs`, so stamping PlateCarree on it
double-transforms and **silently corrupts the basemap** — measured 8,616–34,570 differing pixels.

Installed via matplotlib's documented `_as_mpl_axes()` hook (return `projection`, not the deprecated
`map_projection`). **Not** `register_projection` (dispatches on a string). `add_subplot(projection=...,
axes_class=...)` raises `ValueError`.

Because installation goes through `_as_mpl_axes`, **`Map` never owns the figure**: `Map.axes_projection()` drops
into GridSpec / `add_axes` / `subplots(subplot_kw=)` / insets — which is what `accuml/plot.py:401`
(`plot_predictions`: 3 projected maps + 1 plain scatter panel in one GridSpec) actually needs.

Verified error table (stock `GeoAxes` → `LonLatAxes`): EqualEarth gdf 115–201 px → **0.00**; LAEA gdf 6.3–27.6 px
→ **0.00**; Robinson 116–183 px → **0.00**; LAEA `add_patch` 130.9 → **0.00**; LAEA `text` 10.7–14.0 → **0.00**;
LAEA `da.plot` 7.9e6 → **0.00**. Explicit `transform=` wins and stays silent; `Map(data_crs=None)` restores stock
cartopy.

**Range guard** — a safety net, never proof. Catches every realistic metre CRS (3035, 3857, UTM, Lambert93, even
km-rescaled). **Defeated** by CORDEX rotated-pole grids (`+proj=ob_tran`, standard for regional climate output —
squarely in accuml/wsml's domain), which pass while being wildly misplaced. `gdf.crs` is **provably unreachable**
at the artist level (geopandas discards it before constructing the collection), so the numbers are all plotea can
see. Ship an **xfail** test documenting the hole. Mitigating: unguarded metre data transforms to NaN and the
geometry *silently vanishes* — an empty map is louder than a shifted one; the guard turns disappearance into an
actionable line.

### Logging

Module-level `_log`; a `%(where)s` attribute injected by a **handler-level** stack-walking `Filter` reading
`co_qualname` (3.11+); a **lazy-stdout** handler on the `plotea` logger with `propagate=False`.
Format `%(levelname)s:%(name)s:%(where)s: %(message)s`:

```
INFO:plotea.maps.base:plot_raster: raster has no CRS; assuming lon/lat     <- function
INFO:plotea.maps.base:Map.plot: no transform given; assuming lon/lat       <- method
INFO:plotea.maps.base:Map.europe: preset -> LAEA(10, 52)                   <- classmethod
```

Each choice is forced by a measurement: **Filter** (not LoggerAdapter) keeps the call site a bare `_log.info(...)`;
**`co_qualname`** (not `f_locals['self']`) handles `@staticmethod`; **handler-level** because filters don't
propagate to child loggers, so a filter on `plotea` would miss `plotea.maps.carto`; **lazy stream** because
`StreamHandler(stream=sys.stdout)` — accuml's form at `log.py:28` — binds the object once and escapes
`redirect_stdout` entirely; **`propagate=False`** (not `basicConfig(force=True)`, which rips out the host's
handlers) to stop Jupyter double-printing. `stacklevel=2` is mandatory or the prefix reads `_assume_lonlat`.

This is where plotea beats `accuml/log.py:29` — `%(funcName)s` knows `plot`, never `Map`.

### Styles

`plotea.mplstyle`: de-indent (hygiene — **the indentation is not a bug**, matplotlib strips whitespace and all 18
keys parse), **drop `figure.figsize`** (no fixed value serves both aspects; `Map.plot` derives it, capped ~10 in),
fix the font chain (`font.family: Arial` is a single-element list with no fallback).

`nature.mplstyle` (new): `font.family: sans-serif` + `font.sans-serif: Helvetica, Arial, Nimbus Sans, Liberation
Sans, DejaVu Sans` (Helvetica is **absent** here, Arial present; Nimbus/Liberation are metric-compatible clones);
`font.size: 8` fixed in points; white facecolors explicit; `legend.frameon: False`, spines off; **all `*.linewidth`
≥ 1.0** — mpl 3.11 defaults *violate* Nature (axes 0.8, grid 0.8, tick minor 0.6).

**Guidelines that cannot be rcParams — this list justifies the library:** panel labels (`panel_labels`),
call-site linewidths (`audit_linewidths` — rcParams is blind to accuml's hardcoded 0.4/0.3/0.12), thousands
separator (`colorbar(thousands=True)`), rainbow denylist (`image.cmap` sets a default, it can't stop
`cmap='jet'`). Scale bars **deferred** — accuml has none, so there's no call site to design against.

### Packaging

Add to `pyproject.toml`: `requires-python = ">=3.11"` (`co_qualname`); deps `matplotlib>=3.9,<4` (upper bound
because `_set_artist_props` is private), `cartopy>=0.23`, `geopandas>=1.0`, numpy/xarray/shapely/pyproj; extras
`raster = ["rioxarray"]` (optional — drags in GDAL; consulted behind `try/except ImportError`), `dev`, `docs`;
`[tool.setuptools.packages.find] include = ["plotea*"]`. `environment.yml` stays the conda dev environment and
needs no change; keep the two consistent by hand.

## Incremental delivery

Each commit is reviewable alone, ends green, leaves the tree importable.

### ▶ STEP 1 — pip-installable + a world map. **Then STOP for review.**

The only step authorised now. After this commit:

```python
map = Map()
fig, ax = map.plot()     # whole world, country outlines, Equal Earth, offline
```

Deliberately **no `LonLatAxes` yet** — `Map().plot()` alone doesn't need it, so the axes subclass and its three
hooks move to step 2. `plot()` returns a stock cartopy `GeoAxes` for now; swapping the axes class later is
internal and invisible to this API.

Files:
- `pyproject.toml` — deps, extras, `requires-python`, `[tool.setuptools.packages.find] include = ["plotea*"]`.
  Without this `pip install -e .` ships nothing.
- `plotea/__init__.py`, and an `__init__.py` for `maps/`, `generic/`, `graphs/`, `tseries/`, `volumes/` — none
  exists today, so no subpackage is importable.
- `plotea/log.py` — the full logging design (needed immediately: `Map.plot` is the first thing that logs).
- `plotea/maps/registry.py` — `EXTENTS`, `REGIONS`, `BasemapStyle`, `BASEMAP_PLAIN`/`BASEMAP_MUTED`. Pure data.
- `plotea/maps/utils.py` — `equal_earth()`, `laea(lon, lat)`, `europe_laea()`.
- `plotea/maps/carto.py` — `draw_basemap()`. **`scale='50m'`**, not cartopy's 110m default (see below).
- `plotea/maps/base.py` — `Map`, `Map.plot`, `Map.axes_projection`, aspect-derived figsize.
- `PLAN.md` + `docs/design-maps.md` — this plan and the full verified design, in the repo.
- `tests/test_map.py`, `tests/test_log.py`, `tests/test_offline.py`; `examples/01_hello_map.py`.

**Country outlines — no download script; it is built in.** Verified:
- `cfeature.BORDERS` *is* Natural Earth (`ne_*_admin_0_boundary_lines_land`). Use it directly.
- cartopy **bundles no data** (`repo_data_dir` contains zero shapefiles); it fetches on demand into
  `~/.local/share/cartopy`.
- cartopy **ships a pre-fetch CLI**: `cartopy_feature_download physical cultural` (already on PATH in the
  `plotea` env; `--output` to relocate, `--dry-run` to list URLs). Document this for CI/fresh boxes — do **not**
  write a custom downloader.
- geopandas' `naturalearth_lowres` was **removed in 1.0** (raises `AttributeError`); `geodatasets` not installed.
  That route is dead.
- Cache audit justifies the default: all four **50m** layers (land, ocean, coastline, admin_0 borders) are
  **cached** → `Map()` renders offline today. **110m is missing `ocean` and `coastline`**, so cartopy's own
  default would download. Hence `scale='50m'`, pinned by `test_offline`.

Proof to show you at review: `pip install -e .` succeeds, `Map().plot()` renders the world offline, tests pass.

### Steps 2+ — not authorised yet, listed so the shape is visible

2. **`LonLatAxes` + the three hooks** → `gdf.plot(ax=ax)` works. `_SKIP`, range guard, `_assume_lonlat`,
   `test_transform.py`, `test_basemap_pixel_identity`, the contract canaries.
3. **Styles** — edit `plotea.mplstyle`, add `nature.mplstyle`, `style.py`, `checks.py`; move `styles/` under
   `plotea/`. Decide `BasemapStyle.linewidth` 0.5 vs Nature's 1.0 here.
4. **`save()` + `__init__` re-exports.** ← second review gate; above is hard to reverse, below is leaves.
5. **`colorbar()`** — one policy (`ax.inset_axes`), replacing accuml's six.
6. **`ColorScale`, `continuous`, `diverging`, `class_scale`.**
7. **`plot_raster(da, ax=...)`** — pcolormesh default.
8. **`panel_labels()` + `inset()`.**
9. **The refusal list** in `README.md`.
10. **Adoption proof** — `examples/` + `tests/test_callsites.py` reconstructing accuml's call sites against
    synthetic data **without importing accuml**. Acceptance: each reconstruction is **shorter than the original**.

**Subagents:** each step runs as a **write → review** pair. The reviewer's brief is objective, not taste — the
eight duplications found in accuml: six colorbar mechanisms, two competing basemaps, three hardcoded Europe
extents, the 3×-copy-pasted rcParams block, the 2/98 percentile idiom, the discrete scale built by hand three
times, the hand-rolled figure-coordinate legend, the pixel→degree tick hack. It also enforces the conventions
(arguments on one line, docstring shape, `_log` usage) and the refusal list.

**Retire the payoff risk early:** throwaway-reconstruct `plot_predictor_map` right after step 5. If it isn't
obviously shorter, the design is wrong and it's cheap to learn then. Target (44 lines → 5):

```python
def plot_predictor_map(df, col, ax=None):
    fig, ax = Map(region='europe', extent='europe_wide').plot(ax=ax)
    sc = ax.scatter(df.longitude, df.latitude, c=df[col], s=1, **continuous(df[col]).kw)
    colorbar(sc, ax=ax, label=col)
    return fig, ax
```
Gone: `transform=`, `fig.add_axes([0.84, 0.15, 0.02, 0.60])`, the 2/98 percentile block, the extent literal.

## Verification

plotea proves itself standalone — accuml is not modified.

- **`tests/test_transform.py` — the crown jewel.** {EqualEarth, LAEA, Robinson, PlateCarree} × {gdf points, lines,
  polygons, `add_patch`, `text`, `da.plot`}: plain `gdf.plot(ax=ax)` must land within 1e-6 px of the explicit
  `transform=` reference. **Also assert the bare `GeoAxes` case is non-zero** — otherwise the test cannot fail.
  Plus `test_override_wins` (EPSG:3035 data + explicit transform → 0.00 px, **no log**) and `test_data_crs_none`.
- **`test_basemap_pixel_identity`** — render a 50m basemap on `LonLatAxes` vs stock `GeoAxes`, diff RGBA buffers,
  assert **0 differing pixels**. Unguarded this reads 8,616–34,570. *The most valuable test in the increment* —
  the only thing between you and a silently corrupted basemap on a cartopy upgrade.
- **Test methodology is load-bearing**: use `get_offset_transform()` for point artists, `get_transform()`
  otherwise (`scatter` sets `offset_transform` then `set_transform(IdentityTransform())` — a naive test reports
  false ~516 px failures). Never probe (0,0) — a fixed point of EqualEarth gives a false pass. Never assume vertex
  order — geopandas reorders polygon rings (phantom 63 px error).
- **`test_offline`** — monkeypatch `NEShpDownloader.acquire_resource`, assert `Map().plot()` never enters it; pins
  the 50m default.
- **`test_log.py`** — method/classmethod/staticmethod/function prefixes; lazy-stdout under `redirect_stdout`; no
  double-print with a pre-installed root handler.
- **`test_guard.py`** — 4326 → INFO; `.to_crs(3035)` → WARNING; **xfail** pinning the CORDEX rotated-pole hole.
- **`examples/01_hello_map.py`** — the mandated chain, four lines.
- **Step 6 pins against accuml's real output** (the only behaviour-preservation proof available):
  `class_scale(values=WORLDCOVER_VALUES)` reproduces `worldcover_style`'s bounds `[9.5, 15.0, 25.0, 35.0, ...]`
  exactly; `class_scale(bounds=TN_BOUNDS)` reproduces `tn_class_cmap`'s `(cmap, norm)` exactly.

Run: `/home/kajetan/miniconda3/envs/plotea/bin/python -m pytest tests/ -v`

## Open risks

1. **`_set_artist_props` is private matplotlib API — the biggest risk.** No stability guarantee; failure mode is
   *silent misplacement*, not an exception. There is **no public choke point** (verified). Mitigations: pin
   `<4`, the contract canary, the pixel tests. Hedge: **also** override public `add_collection` as belt-and-braces
   so the gdf path — the one you actually mandated — degrades gracefully if the private method vanishes.
2. **Cartopy's decorated set is version-coupled** (11 in 0.25). A 12th = silent misplacement. The canary is the
   defence. Note my own briefing drifted from reality on this list — treat inherited facts as unverified.
3. **`text` was found by inspection, not systematic search.** Audit `bar`, `errorbar`, `arrow`, `fill_between`
   before v1.
4. **`LonLatAxes` is a `GeoAxes` that behaves differently from a `GeoAxes`.** Pasted cartopy examples silently
   change meaning — correctly, far more often than not, but the surprise is real, and the log fires once per axes
   so a 50-panel loop shows one easy-to-miss line. This is the direct cost of your transform decision, and your
   `gdf.plot(ax=ax)` requirement cannot be honoured any other way. **Document as a headline, not a footnote.**
5. **The range guard is defeatable** (CORDEX rotated-pole). Ship the xfail as documentation.
6. **Nature's ≥1 pt rule collides with dense maps.** `plot_tn_map` uses `edgewidth=0.12` on thousands of points
   *because* 1 pt makes a black blob. `audit_linewidths` must **log, never raise**. Unresolved:
   `BasemapStyle.linewidth` 0.5 (screen) vs Nature 1.0 — **decide at step 3**.
7. **Offline is a property of this machine's cache**, not of cartopy — cartopy bundles no data. CI must run
   `cartopy_feature_download physical cultural` or mark network tests.
   *Disclosure*: verification probing fetched `ne_110m_land.*` into `~/.local/share/cartopy` (mtime 2026-07-17).
   No repo file touched.
8. **Geopandas internals are an implementation detail** — `gdf.plot(ax=ax)` works because `**style_kwds` are
   forwarded into `PatchCollection`/`scatter`. Not a documented contract. Pin `geopandas>=1.0`.
9. **`method='imshow'` in `plot_raster` is offered but unverified** — don't claim it until tested.

**Deliberately refused** (ship in README at step 9 — the cheapest durable guard on the scope boundary): SHAP/PDP/
obs-vs-pred/correlation (`accuml/plot.py:351,473,678,825`); TN + WorldCover semantics (`:179,:912-915,:978`);
contextily (`:233`); `Raster`/`Vector` classes; rasterio IO; scale bars; anything in `plotea/legacy/`;
`graphs/`, `tseries/`, `volumes/`. Step 10's examples are the tripwire: if an example needs a TN constant *from
plotea*, the boundary has been crossed.
