# 🍃 plotea

Publication-quality maps, with the choices a map needs made for you: the projection from
your data, a scale bar that is the right length, a colorbar that fits the axes, and panels
that stay aligned.

Cartopy projects axes and draws coastlines. plotea composes the figure.

![A map of Sardinia with a wider view and a closer one beside it](docs/_static/sardinia_panels.png)

That figure is one screen of code, from a digital elevation model that ships with the
repository — see [`notebooks/plotea.ipynb`](notebooks/plotea.ipynb).

## Installation

```bash
git clone git@github.com:kmch/plotea.git && cd plotea && pip install -e .
```

There is a conda build, `conda install kmch::plotea`, but it is **0.0.2, from before this
rewrite**: it carries the old `plotea.mpl2d` modules and none of the map package described
here. Until a new build is pushed, the editable install above is the only way to get this
version.

The HEALPix layer needs two packages of its own and is the one part a map-only install has
no use for, so it is an extra: `pip install -e '.[dggs]'`. Likewise `.[dev]` for the test
suite and `.[docs]` to build the documentation.

## Quick start

```python
from plotea import BaseMap

fig, ax = BaseMap().plot()            # the whole world, country outlines, Equal Earth
```

A `BaseMap` is a specification, not a drawing: constructing one records your choices and
draws nothing until `plot()` is called, so one specification can be drawn into several
figures.

For a raster, the specification can come from the data:

```python
from plotea import BaseMap, Raster, land_terrain, scalebar

dem = Raster('elevation.tif', label='Elevation', unit='m')

fig, ax = BaseMap(**dem.basemap_options()).plot()
dem.plot(ax=ax, cmap=land_terrain(), blend_mode='hsv', vert_exag=3, cbar=True)
scalebar(ax, length_km=50)
```

`basemap_options()` takes the view from the raster's own extent and the projection from its
centre — a Lambert azimuthal equal-area, so areas stay comparable — and leaves the graticule
step to cartopy, which reads it off the extent. A fixed continental projection would draw a
raster from elsewhere in the world distorted or off the frame; a fixed graticule step draws
nothing on a small region and a solid mesh on a large one.

And before trusting a file, look at it:

```python
dem.details        # every gdalinfo field in one labelled column, value range computed exactly
```

## What is here

| | |
| --- | --- |
| `Raster` | read, decimate, shade, reproject, clip, align to a grid, describe. Cartopy has no raster object. |
| `Raster.details` | every `gdalinfo` field in one labelled column, value range computed exactly |
| `basemap_options` | the projection and graticule derived from the data rather than named |
| `blend_mode` | colour blended with relief in one call, with a working colorbar |
| `scalebar` | segmented, true-length, drawn in projected coordinates |
| `Colorbar` | height matched to a fixed-aspect map axes, formatter fixed once so panels agree |
| `LonLatAxes` | bare degrees are degrees, so a patch or a scatter cannot land in metres |
| `BaseMap` | a specification that draws nothing until asked, reusable across figures |
| `panel_layout` | a main map beside aligned panels, placed as rectangles so equal-aspect axes cannot drift |
| `Bbox` | a view as a name instead of four numbers, and it draws itself |
| `Swath` | values on a curvilinear sensor grid, which `imshow` cannot place |
| `HealpixCells` | discrete global grid cells as filled quadrilaterals, not points |

Reading a raster is decimated by default: `max_px` caps the longest side of what is read, so
a continental mosaic gives a 2000-pixel preview instead of 40 GB — the difference between a
figure you can iterate on and one you run overnight.

## Why the defaults are what they are

Every default answers to a journal's figure requirements rather than to taste: sans-serif
lettering, no line thinner than one point, lower-case bold panel labels, a scale bar rather
than a magnification factor, a white background. The requirements are quoted in
[`docs/figure-guidelines.md`](docs/figure-guidelines.md), along with the two plotea does
**not** yet meet — there is no colour-blind-safe sequential ramp on offer, and nothing groups
thousands with commas.

## Status

Version 0.1.0, and the honest summary is that the map package is what works and what is
tested (68 tests). The `description` in `pyproject.toml` still promises graphs, time series
and volumes, which are not here. `import plotea` pulls every public name from every module
into the top level for convenience; a curated public surface comes later.

## Citing

If you use any part of this repository in your research, please cite:

- Chrapkiewicz, Kajetan and Lipp, Alex and Barron, Leon Patrick and Barnes, Richard and
  Roberts, Gareth, **Apportioning sources of chemicals of emerging concern along an urban
  river with inverse modelling**. Science of the Total Environment.
  https://doi.org/10.1016/j.scitotenv.2024.172827
