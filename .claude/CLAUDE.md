# plotea — project conventions

plotea is a library for publication-quality scientific figures (maps, graphs, time
series, volumes), built on matplotlib and cartopy. These rules apply to every
session and every subagent working in this repo.

## Hard rules

1. **Never name external consumer packages** in code, comments or docstrings. plotea
   is a standalone library; the projects that will adopt it must not appear in its
   source. If you need to justify a design with a real call site, describe the
   *pattern* generically ("a scatter-on-map with a fixed colorbar"), not the package.
2. **`legacy/` is frozen.** Do not edit, import, re-export or "fix" anything under
   `plotea/legacy/`. It is excluded from the installed package.
3. **Deliver in small, reviewable steps.** One concern per commit; the tree stays
   importable and the tests stay green after every step.

## Naming and terminology

- **`crs`**, never `projection`, for a cartopy coordinate reference system.
- **`resolution`**, never `scale`, for Natural Earth detail ('50m', '110m', '10m').
- **`extent`** is a *view* — `[lon_min, lon_max, lat_min, lat_max]` in degrees, or
  `None` for the whole world. It is not the CRS.
- The basemap class is **`BaseMap`**; instances are `bm`.
- Default world CRS is **Equal Earth** (equal-area); Europe's preset is **LAEA**
  centred on (10, 52).

## Docstrings (numpydoc)

- `"""` always on its own line; a blank line above the closing `"""`.
- Always include an `Examples` section with runnable usage.
- In prose, use **double backticks** ``` ``x`` ``` for code (class names, params,
  values, modules) — that renders as inline literal. Single backticks are RST
  "interpreted text" (italic / cross-reference), not code.
- Numpy sections: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`.

## Code style

- **Arguments on one line**: `def foo(a: int, b: int) -> T:` — never wrap the
  parameter list across lines.
- **Logging**: module-level `_log = get_logger(__name__)`, used as `_log.info('...')`
  in both plain functions and methods. In a class method the prefix automatically
  becomes `Class.method`. Output must be visible in Jupyter; switch it on with
  `plotea.set_log_level()`.
- Prefer a plain module-level function over a `@staticmethod` when the logic needs
  neither instance nor class state.
- Keep cartopy imports confined to `plotea/maps/crs.py` and `plotea/maps/carto.py`.

## Module map (maps)

- `crs.py` — CRS constructors and `resolve_crs`. The only place CRSs are named.
- `vector.py` — the `Vector` wrapper, the `Bbox` view (padded box from geometry,
  `.extent` for cartopy, `.plot` for the outline), and `resolve_bbox`/`named_bbox`.
- `registry.py` — `ROIS`, named regions of interest a string bbox resolves to.
- `styles.py` — `BasemapStyle` and the ready-made styles.
- `carto.py` — the cartopy backend: `new_axes`, `draw_basemap`.
- `base.py` — `BaseMap`.

## Environment and verification

- Interpreter: `/home/kajetan/miniconda3/envs/plotea/bin/python` (the `plotea` conda
  env: python 3.13, matplotlib, cartopy, geopandas, xarray; no rasterio).
- Headless rendering: set `MPLBACKEND=Agg`.
- Run tests: `python -m pytest tests/ -q`.
- A change to plotting code is not done until it has been *rendered* and looked at,
  not merely unit-tested.
