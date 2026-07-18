"""
plotea: publication-quality scientific figures -- maps, graphs, time series, volumes.

Notes
-----
The core contract is::

    import plotea
    plotea.set_log_level()
    bm = plotea.BaseMap()
    fig, ax = bm.plot()          # whole world, country outlines, Equal Earth

For now this pulls every public name from every module into the top-level
namespace, for convenience. A curated public surface comes later.

Legacy code under ``plotea.legacy`` is frozen and not re-exported.

"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotea.log import *
from plotea.maps.crs import *
from plotea.maps.registry import *
from plotea.maps.vector import *
from plotea.maps.basemap_styles import *
from plotea.maps.carto import *
from plotea.maps.base import *
