"""
plotea: publication-quality scientific figures -- maps, graphs, time series, volumes.

Notes
-----
The core contract is::

    import plotea
    bm = plotea.BaseMap()
    fig, ax = bm.plot()          # whole world, country outlines, Equal Earth

Logging is switched on automatically at import (INFO), so log lines show in a
notebook with no extra call; use ``plotea.init_logging(plotea.WARNING)`` to quiet
it or ``plotea.init_logging(plotea.DEBUG)`` for more.

For now this pulls every public name from every module into the top-level
namespace, for convenience. A curated public surface comes later.

Legacy code under ``plotea.legacy`` is frozen and not re-exported.

"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotea.log import *
from plotea.generic.scales import *
from plotea.generic.labels import *
from plotea.generic.scalebar import *
from plotea.maps.crs import *
from plotea.maps.registry import *
from plotea.maps.vector import *
from plotea.maps.styles import *
from plotea.maps.carto import *
from plotea.maps.base import *
from plotea.maps.layout import *
from plotea.maps.raster import *

init_logging()
