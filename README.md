# 🍃 plotea

Publication-quality maps, with the choices a map needs made for you: the projection taken
from your data, a scale bar at the right length, a colorbar that fits the axes, panels that
stay aligned.

It wraps `matplotlib` and `cartopy` for drawing and `rasterio`/`rioxarray` for reading, and
complements cartopy rather than replacing it — cartopy projects axes and draws coastlines,
plotea composes the figure.

![A map of Sardinia with a wider view and a closer one beside it](docs/_static/sardinia_panels.png)

One screen of code, on a DEM that ships with the repository. See
[`notebooks/plotea.ipynb`](notebooks/plotea.ipynb) for that figure and everything else.

## Installation

```bash
git clone git@github.com:kmch/plotea.git && cd plotea && pip install -e .
```

Extras: `.[dggs]` for the HEALPix layer, `.[dev]` for the tests, `.[docs]` to build the
documentation. There is a conda build, `conda install kmch::plotea`, but it is 0.0.2 from
before this rewrite and has none of the map package.

## Quick start

```python
from plotea import BaseMap, Raster, land_terrain, scalebar

fig, ax = BaseMap().plot()                         # the world, country outlines, Equal Earth

dem = Raster('elevation.tif', label='Elevation', unit='m')

fig, ax = BaseMap(**dem.basemap_options()).plot()  # view and projection from the raster
dem.plot(ax=ax, cmap=land_terrain(), blend_mode='hsv', cbar=True)
scalebar(ax, length_km=50)

dem.info                                 # what the file is; .details adds the value range
window = dem.read_window(max_px=2000)    # or every=10, a stride; a DataArray with its own footprint
window.plotea.extent                     # cell edges, ready for imshow
```

## Citing

If you use any part of this repository in your research, please cite:

- Chrapkiewicz, Kajetan and Lipp, Alex and Barron, Leon Patrick and Barnes, Richard and
  Roberts, Gareth, **Apportioning sources of chemicals of emerging concern along an urban
  river with inverse modelling**. Science of the Total Environment.
  https://doi.org/10.1016/j.scitotenv.2024.172827
