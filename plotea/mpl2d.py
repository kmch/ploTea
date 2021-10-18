"""
Wrapper around matplotlib 2d plotting 
capabilities. 
"""
from abc import ABC, abstractmethod
import cmocean
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource #, LogNorm
import numpy as np

from plotea.generic import FigFactory, Fig

@FigFactory.register_subclass('mpl2d')
class FigMpl(Fig):
  """
  Matplotlib figure with 2d axes.

  Notes
  -----
  This object-oriented interface serves to 
  create finely-tuned, complex figures.

  For quick plots on an existing figure,
  just use the standalone plot() function.
  """
  def __init__(self, xsize=6, ysize=6, nrows=1, ncols=1,\
    projection='2d', style='fwipy'):
    """
    """
    self._set_size(xsize, ysize)
    self._set_style(style)
    
    self.fig = plt.figure(figsize=self.figsize)
    self._init_all_axes(nrows, ncols)
    # self._create_all_axes()
    # self._set_current_axes()
    self.cax = plt.gca()
  def plot_image(self, image):
    return self.cax.imshow(image)  
  def plot_line(self, y):
    return self.cax.plot(y)
  def save(self, *args, **kwargs):
    """
    Format is deduced from the fname.

    """
    plt.savefig(*args, **kwargs)
  # -----------------------------------------------------------------------------
  def _add_all_axes(self, nrows, ncols, projs='2d'):
    """
    """
    from matplotlib.gridspec import GridSpec
    self.gs = GridSpec(nrows, ncols)
    
    for x in range(nrows):
      for y in range(ncols):
        self.axes[x][y] = self._add_axes2d(x, y, self.projection[x,y])
  def _add_axes_2d(self, x, y, projection):
    self.axes[x][y] = self.fig.add_subplot(self.gs[x,y],\
       projection=projection)
  def _add_layout(self):
    from matplotlib.gridspec import GridSpec
    self.gs = GridSpec(nrows, ncols)
  def _create_axes(self, nrows, ncols, projection='2d'):
    """
    You can later set_height_rat.shell etc.
    
    # cax = fig.add_subplot(111) # 111 needed, otherwise
    # ... returns None

    """
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    
    self.gs = GridSpec(nrows, ncols)
    self._set_axes_projections(projection, nrows, ncols)

    self.axes = [] # ARRAY CAN'T STORE SUBPLOTS APPARENTLY
    for x in range(nrows):
      tmp = []
      for y in range(ncols):
        tmp.append(\
          self.fig.add_subplot(self.gs[x,y], projection=self.projection[x,y]))
      self.axes.append(tmp)
  def _init_all_axes(self, nrows, ncols):
    self.axes = np.zeros((nrows, ncols)).tolist()
    # self._set_axes_projections(projection, nrows, ncols)  
  def _set_axes_projections(self, projection, nrows, ncols):
    """
    """
    if isinstance(projection, str):
      projection = np.full((nrows, ncols), projection)
    elif isinstance(projection, np.arrauray):
      assert projection.shape == (nrows, ncols)
      projection = projection
    else:
      raise TypeError('type(projection)', type(projection))
    self.projection = np.where(projection=='2d', None, projection)
  def _set_current_axes(self, x=0, y=0):
    self.cax = self.axes[x][y]
  def _set_size(self, xsize, ysize):
    self.figsize = (xsize, ysize)
  def _set_style(self, style):
    plt.style.use(['default', 'ggplot', style])
# -------------------------------------------------------------------------------
class PlotterMpl(ABC):
  def plot(self, arr, **kwargs):
    self._prep(**kwargs)
    self._plot(arr)
    self._format(**kwargs)
  # -----------------------------------------------------------------------------    
  def _format(self, **kwargs):
    """
    Kwargs have to be the original 
    ones fed into plot().
    """
    self.ax.set_aspect(kwargs.get('aspect', 'equal'))
    self.ax.set_xlabel(kwargs.get('xlabel', None))
    self.ax.set_ylabel(kwargs.get('ylabel', None))
    self._invert_vertical_axis(self.ax, **kwargs)
  def _invert_vertical_axis(self, ax, **kwargs):
    pass
  def _prep(self, **kwargs):
    self.ax = kwargs.get('ax', plt.gca())
    self._parse_kwargs(**kwargs)  
  # -----------------------------------------------------------------------------
  @abstractmethod
  def _parse_kwargs(self, **kwargs):
    """
    From all kwargs, return only those understood
    by the plotting function.
    """
    pass    
  @abstractmethod
  def _plot(self, arr):
    pass
class Contour(PlotterMpl):
  def _invert_vertical_axis(self, ax, **kwargs):
    if not kwargs.get('vertical_axis_up', True):
      ax.invert_yaxis()  
  def _plot(self, arr):
    self.ax.contour(arr.T, **self.kwargs)
  def _parse_kwargs(self, **kwargs):
    new_kwargs = {}
    defaults = dict(extent=None, levels=10, colors='k', linewidths=.4)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs = new_kwargs  
class Contourf(Contour):
  def _plot(self, arr):
    self.ax.contourf(arr.T, **self.kwargs)
  def _parse_kwargs(self, **kwargs):
    new_kwargs = {}
    defaults = dict(extent=None, levels=10)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs = new_kwargs      
class Imshow(PlotterMpl):
  def _invert_vertical_axis(self, ax, **kwargs):
    if kwargs.get('vertical_axis_up', True):
      # print('Inverting vertical axis.')
      ax.invert_yaxis()    
  def _plot(self, arr):
    im = self.ax.imshow(arr.T, **self.kwargs_imshow)
    self.ax.figure.colorbar(im, ax=self.ax, **self.kwargs_cbar)
  def _parse_kwargs(self, **kwargs):
    self._parse_kwargs_imshow(**kwargs)
    self._parse_kwargs_cbar(**kwargs)  
  def _parse_kwargs_imshow(self, **kwargs):
    new_kwargs = {}
    defaults = dict(cmap=None, norm=None, aspect=None, interpolation=None, \
      alpha=None, vmin=None, vmax=None, origin=None, extent=None)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_imshow = new_kwargs
  def _parse_kwargs_cbar(self, **kwargs):  
    new_kwargs = {}
    defaults = dict(label='')
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_cbar = new_kwargs      
class PltPlot(PlotterMpl):
  """
  Wrapper around plt.plot
  """
  def _plot(self, arr):
    if self.xaxis is None:
      self.args = [arr]
    else:
      self.args = [self.xaxis, arr]
    plt.plot(*self.args, **self.kwargs)
  def _parse_kwargs(self, **kwargs):
    new_kwargs = {}
    defaults = dict(color='k', linestyle='-', marker=None)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs = new_kwargs
  def _prep(self, **kwargs):
    super()._prep(**kwargs)
    self.xaxis = kwargs.get('xaxis', None)
class Shade(Imshow):
  def _parse_kwargs(self, **kwargs):
    super()._parse_kwargs(**kwargs)
    self._parse_kwargs_light_source(**kwargs)
    self._parse_kwargs_shade(**kwargs)
  def _parse_kwargs_light_source(self, **kwargs):  
    new_kwargs = {}
    defaults = dict(azdeg=45, altdeg=45)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_light_source = new_kwargs    
  def _parse_kwargs_shade(self, **kwargs):
    new_kwargs = {}
    defaults = dict(blend_mode='soft', vert_exag=1)
    # convert string into colormap as required by shade
    new_kwargs['cmap'] = plt.cm.get_cmap(self.kwargs_imshow.get('cmap', None))    
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_shade = new_kwargs
  def _plot(self, arr):
    imshow_mappable = self.ax.imshow(arr.T, **self.kwargs_imshow)
    im = LightSource(**self.kwargs_light_source).shade(arr.T, **self.kwargs_shade)
    im = self.ax.imshow(im, **self.kwargs_imshow)
    self.ax.figure.colorbar(imshow_mappable, ax=self.ax, **self.kwargs_cbar)
# -------------------------------------------------------------------------------
class Scroller(ABC):
  """
  Redraw axes upon
  mouse-scrolling action.
  
  Notes
  ------
  Usage in jupyter (outdated)
  >>> %matplotlib notebook
  >>> fig = plt.figure(figsize=[6,12])
  >>> tracker = p01.i.tvp.scrollall(fig, istep=5, cmap='cividis')
  >>> fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

  """
  def onscroll(self, event):
    if event.button == 'up':
      self.ind = (self.ind + self.istep) % self.svalue_max
    else:
      self.ind = (self.ind - self.istep) % self.svalue_max
    self.update()

  @abstractmethod
  def __init__(self):
    # plot the initial states
    self.update()

  @abstractmethod
  def update(self):
    pass
# -------------------------------------------------------------------------------
# Basic functions
# -------------------------------------------------------------------------------
def plot(arr, ax, **kwargs):
  nd = len(shape(arr))
  if  nd == 1:
    func = plot_array_1d
  elif nd == 2:
    func = plot_array_2d
  else:
    raise TypeError('Wrong no. of arr dimensions: %s' % nd)
  func(arr, ax, **kwargs)
def plot_array_2d(arr, **kwargs):
  """
  Framework for 2d plots.

  Parameters
  ----------
  arr : 2d arrauray or Arr2d
  """
  return Imshow.plot(arr, **kwargs)
# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------
def aspeqt(ax=plt.gca(), **kwargs):
  ax.set_aspect('equal')
def autect(ax=plt.gca(), **kwargs):
  ax.set_aspect('auto')
def cat_cmaps(cmaps, vmin, vmax):
  """
  Combine 2 colormaps.
  
  Parameters
  ----------
  FIXME
  """    
  if len(cmaps) == 2:
    cmap1, cmap2 = cmaps
  elif len(cmaps) == 0:
   cmap1 = 'cmo.turbid'
   cmap2 = 'cmo.ice_r'    
  else:
    raise ValueError('len(cmaps): %s' % len(cmaps))
  
  cmap1 = plt.cm.get_cmap(cmap1)
  cmap2 = plt.cm.get_cmap(cmap2)
  
  n = 100
  ratio = np.abs(vmin)/np.abs(vmax)
  # print('ratio',ratio)
  colors1 = cmap1(np.linspace(0., 1, int(ratio*n)))
  colors2 = cmap2(np.linspace(0, 1, n))
  colors = np.vstack((colors1, colors2))
  my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
  return my_cmap
def cm2inch(value, unit):
  inch = 2.54 # cm
  if unit == 'cm':
    # convert to inches
    value /= inch
  elif unit == 'inch':
    pass # because mpl expects inches
  elif unit != 'inch':
    raise ValueError('unit: %s'  % unit)  
  return value
def figure(figsize_x=6, figsize_y=6, unit='inch', **kwargs):
  """
  Apparently one has to create a new figure INSIDE 
  a function passed to interact. 
  This is the code that has to be put in every 
  function decorated with ##@widgets then.
  """
  figsize_x = cm2inch(figsize_x, unit)
  figsize_y = cm2inch(figsize_y, unit)  
  figsize = (figsize_x, figsize_y)
  return plt.figure(figsize=figsize)
def figax(figsize_x=15, figsize_y=5, unit='inch', **kwargs):
  """
  A4	210 x 297 mm,	8.3 x 11.7 inches
  """
  figsize_x = cm2inch(figsize_x, unit)
  figsize_y = cm2inch(figsize_y, unit)
  figsize = (figsize_x, figsize_y)
  fig, ax = plt.subplots(figsize=figsize)
  return fig, ax 
def flipy(ax=plt.gca(), **kwargs):
  ax.invert_yaxis()
def set_xlabels(labels, decim_xlabels=10, rotate_xlabels=None, **kwargs):
  """
  Decimate and rotate labels.

  labels : list 
      Labels before decimation
  """  
  locs = np.arange(len(labels))[::decim_xlabels]
  labels = labels[::decim_xlabels]
  rotation = lambda decim : np.clip(90 - 10 * (decim - 1), 0, 90)
  if rotate_xlabels is None:
    rotate_xlabels = rotation(decim_xlabels)
  set_xlabels._log.debug('Rotating xlabels %s degrees' % rotate_xlabels)
  locs, labels = plt.xticks(locs, labels, rotation=rotate_xlabels)
  return locs, labels
# -------------------------------------------------------------------------------
