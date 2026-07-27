from plotea.log import get_logger
from plotea.maps.registry import ROIS

_log = get_logger(__name__)

class Raster:
    """
    Generic raster object.

    Can be initialised with a file path, an xarray DataArray, both, or neither.
    Data is loaded from disk lazily — only when first accessed via `.data`.


    Parameters
    ----------
    path : str or Path, optional
    data : xarray.DataArray, optional
    """
    def __init__(self, path=None, data=None):
        self.path = Path(path) if path is not None else None
        self._data = data
    def __repr__(self):
        """
        Return a string representation of the Raster object, showing the path and
        whether the associated data has been loaded into memory.

        Returns
        -------
        str
            String of the form "Raster(path=..., data=loaded)" if data is loaded,
            or "Raster(path=..., data=not loaded)" if not.
        """
        data_status = 'loaded' if self._data is not None else 'not loaded'
        return f"Raster(path={self.path}, data={data_status})"
    def info(self):
        print(f"path      : {self.path}")
        print(f"has data  : {self._data is not None}")
        if self.data is not None:
            print(f"shape     : {self.data.shape}")
            print(f"dtype     : {self.data.dtype}")
            print(f"crs       : {self.crs}")
            print(f"bounds    : {self.bounds}")
            print(f"resolution: {self.resolution}")

    # Data access (lazy) ------------------------------------------------------------------
    @property
    def data(self) -> xr.DataArray | None:
        if self._data is None and self.path is not None:
            self._data = rioxarray.open_rasterio(self.path, masked=True)
        return self._data
    # This is the setter for the 'data' property of the Raster class.
    # It allows assignment like `raster.data = new_data`, which updates the internal
    # _data attribute. This is useful for replacing or injecting new xarray.DataArray
    # content without changing the file path.
    @data.setter
    def data(self, value):
        self._data = value

    # Spatial metadata — derived from data on demand ---------------------------- 
    @property
    def crs(self):
        return self.data.rio.crs if self.data is not None else None
    @property
    def transform(self):
        return self.data.rio.transform() if self.data is not None else None
    @property
    def bounds(self):
        return self.data.rio.bounds() if self.data is not None else None
    @property
    def resolution(self):
        return self.data.rio.resolution() if self.data is not None else None
 
    # Geospatial ------------------------------------------------------------------
    def clip(self, mode, out_file, **kwargs) -> 'Raster':
        """Clip and write to *out_file*. See standalone clip() for full docs."""
        if self.path is None:
            raise ValueError("clip requires a file path (self.path).")
        return clip(self.path, mode, out_file, **kwargs)
    def reproject(self, dst_crs: int, out_file=None) -> 'Raster':
        """Return a new Raster reprojected to *dst_crs* (EPSG int)."""
        if self.path is None:
            raise ValueError("reproject requires a file path (self.path).")
        dst_crs_str = f'EPSG:{dst_crs}'
        if out_file is None:
            out_file = self.path.parent / f'{self.path.stem}_reproj_{dst_crs}.tif'
        out_file = Path(out_file)
        with rasterio.open(self.path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs_str, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(crs=dst_crs_str, transform=transform, width=width, height=height)
            with rasterio.open(out_file, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=transform, dst_crs=dst_crs_str,
                        resampling=Resampling.nearest,
                    )
        _log.info(f"Reprojected to {dst_crs_str}, saved to {out_file}")
        return Raster(path=out_file)

    # Visualisation ----------------------------------------------------------------
    def plot(self, ax=None, cmap='viridis', cbar=True, band=None,
             vmin=None, vmax=None, log_scale=False, title=None, aspect=None, **kwargs):
        if self.data is None:
            raise ValueError("No data to plot.")
        if ax is None:
            ax = plt.gca()
        arr = self.data.sel(band=band) if band is not None else self.data #FIXME?
        norm = LogNorm() if log_scale else None
        arr.plot(ax=ax, cmap=cmap, add_colorbar=cbar, add_labels=False,
                 vmin=vmin, vmax=vmax, norm=norm, **kwargs)
        if title is not None:
            ax.set_title(title)
        if aspect is not None:
            ax.set_aspect(aspect)
        return ax

    # I/O ------------------------------------------------------------------
    def save(self, out_file, driver='GTiff', **kwargs) -> Path:
        """Write `.data` to *out_file*. Returns the output path."""
        if self.data is None:
            raise ValueError("No data to save.")
        out_file = Path(out_file)
        self.data.rio.to_raster(out_file, driver=driver, **kwargs)
        _log.info(f"Saved raster to {out_file}")
        return out_file

