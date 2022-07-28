"""
Wrapper around matplotlib 2d plotting 
capabilities. 
"""
from abc import ABC, abstractmethod
import cmocean
import matplotlib.colors as mcolors # for cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource #, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    if 'xlabel' in kwargs: # otherwise plot_slice_lines overwrites
      self.ax.set_xlabel(kwargs.get('xlabel', None))
    if 'ylabel' in kwargs:
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
    cntr = self.ax.contour(arr.T, **self.kwargs_contour)
    show_cntr_labels = self.kwargs_clabel['show_cntr_labels']
    del self.kwargs_clabel['show_cntr_labels']
    if show_cntr_labels:
      self.ax.clabel(cntr, cntr.levels, **self.kwargs_clabel)
  def _parse_kwargs(self, **kwargs):
    self._parse_kwargs_contour(**kwargs)
    self._parse_kwargs_clabel(**kwargs)
  def _parse_kwargs_contour(self, **kwargs):
    new_kwargs = {}
    defaults = dict(extent=None, levels=10, colors='k', linewidths=.4, linestyles='solid')
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_contour = new_kwargs
  def _parse_kwargs_clabel(self, **kwargs):
    new_kwargs = {}
    defaults = dict(show_cntr_labels=False, fontsize='smaller', fmt='%1.0f')
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_clabel = new_kwargs        
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
  def _add_colorbar(self, mappable):
    divider = make_axes_locatable(self.ax)
    cax = divider.append_axes(**self.kwargs_divider)
    cbar = self.ax.figure.colorbar(mappable, cax=cax, **self.kwargs_cbar) 
    plt.sca(self.ax) # essential
  def _invert_vertical_axis(self, ax, **kwargs):
    if kwargs.get('vertical_axis_up', True):
      # print('Inverting vertical axis.')
      ax.invert_yaxis()
  def _parse_kwargs(self, **kwargs):
    self._parse_kwargs_cbar(**kwargs)
    self._parse_kwargs_divider(**kwargs)
    self._parse_kwargs_imshow(**kwargs)
    self._parse_kwargs_misc(**kwargs)
  def _parse_kwargs_cbar(self, **kwargs):  
    new_kwargs = {}
    defaults = dict(label='', orientation='vertical')
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_cbar = new_kwargs      
  def _parse_kwargs_divider(self, **kwargs):  
    new_kwargs = {}
    defaults = dict(position='right', size='3%', pad=0.2)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_divider = new_kwargs 
  def _parse_kwargs_imshow(self, **kwargs):
    new_kwargs = {}
    defaults = dict(cmap=None, norm=None, aspect=None, interpolation=None, \
      alpha=None, vmin=None, vmax=None, origin=None, extent=None)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_imshow = new_kwargs
  def _parse_kwargs_misc(self, **kwargs):
    new_kwargs = {}
    defaults = dict(cbar=True, center_cmap=False)
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_misc = new_kwargs  
  def _plot(self, arr):
    self._set_cmap(arr)
    mappable = self.ax.imshow(arr.T, **self.kwargs_imshow)
    if self.kwargs_misc['cbar']:
      self._add_colorbar(mappable)
  def _set_cmap(self, arr):
    cmap = self.kwargs_imshow['cmap']
    center_cmap = self.kwargs_misc['center_cmap']
    if isinstance(cmap, list):
      if len(cmap) == 0:
        cmap = ['cmo.turbid', 'cmo.deep']
      else:
        assert len(cmap) == 2
      cm1, cm2 = cmap
      cmap, norm = cat_2_cmaps(arr, cm1, cm2, centre=0)
      self.kwargs_imshow['cmap'] = cmap
      self.kwargs_imshow['norm'] = norm
    elif center_cmap:
      vmin = self.kwargs_imshow['vmin']
      vmax = self.kwargs_imshow['vmax']
      if vmin is None:
        vmin = np.min(arr)
      if vmax is None:
        vmax = np.max(arr)
      vmin, vmax = _center_around_zero(vmin, vmax)
      self.kwargs_imshow['vmin'] = vmin
      self.kwargs_imshow['vmax'] = vmax
class PltPlot(PlotterMpl):
  """
  Wrapper around plt.plot
  plotting 1d lines on 2d canvas.
  """
  def _plot(self, arr):
    if self.xaxis is None:
      self.args = [arr]
    else:
      self.args = [self.xaxis, arr]
    plt.plot(*self.args, **self.kwargs)
  def _parse_kwargs(self, **kwargs):
    new_kwargs = {}
    defaults = dict(color='k', linestyle='-', marker=None, label='', lw=2, alpha=1)
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
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs_shade = new_kwargs
  def _plot(self, arr):
    self._set_cmap(arr)
    # convert string into colormap as required by shade
    self.kwargs_shade['cmap'] = plt.cm.get_cmap(self.kwargs_imshow['cmap'])
    self.kwargs_shade['norm'] = self.kwargs_imshow['norm']
    mappable = self.ax.imshow(arr.T, **self.kwargs_imshow) # get mappable
    shaded = LightSource(**self.kwargs_light_source).shade(arr.T, **self.kwargs_shade)
    im = self.ax.imshow(shaded, **self.kwargs_imshow) # actual plot
    if self.kwargs_misc['cbar']:
      self._add_colorbar(mappable)
class Wiggle(PlotterMpl):
  def _format(self, **kwargs):
    kwargs['aspect'] = kwargs.get('aspect', 'auto')
    super()._format(**kwargs)  
  def _parse_kwargs(self, **kwargs):
    new_kwargs = {}
    defaults = dict(gap=1, c='k')
    for key, val in defaults.items():
      new_kwargs[key] = kwargs.get(key, val)
    self.kwargs = new_kwargs    
  def _plot(self, arr):
    gap = self.kwargs['gap']
    axis0 = 0
    for i, trace in enumerate(arr):
        axis0 = i * gap 
        plt.plot(trace+axis0, c=self.kwargs['c'])    
# -------------------------------------------------------------------------------
class ScrollerMpl(ABC):
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
  def __init__(self):
    self._init()
    self._update()
  def onscroll(self, event):
    if event.button == 'up':
      self.ind = (self.ind + self.istep) % self.svalue_max
    else:
      self.ind = (self.ind - self.istep) % self.svalue_max
    self.update()
  @abstractmethod
  def _init(self):
    pass
  @abstractmethod
  def _update(self):
    pass
class SliceScroller(ScrollerMpl):
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
def cat_2_cmaps(data, cmap1, cmap2, centre=0, N=50):
  # https://stackoverflow.com/questions/30082174/join-two-colormaps-in-imshow
  cmap = {name:plt.get_cmap(name) for name in (cmap1, cmap2)}
  vmin, vmax = (np.nextafter(min(data.min(), -1), -np.inf), 
                np.nextafter(max(data.max(), 1), np.inf))
  levels = np.concatenate([np.linspace(vmin, centre, N, endpoint=False),
                           np.linspace(centre, vmax, N+1, endpoint=True)])
  colors = np.concatenate([cmap[name](np.linspace(0, 1, N)) 
                           for name in (cmap1, cmap2)])
  cmap, norm = mcolors.from_levels_and_colors(levels, colors)
  return cmap, norm
def _center_around_zero(minn, maxx, **kwargs): #NOTE
  """
  Center a diverging colormap around zero.
  
  Parameters
  ----------
  minn, maxx : float
    Extreme value of the image tfullwavepy.plot.
  
  **kwargs : keyword arguments (optional)
    Current capabilities: 
  
  """  
  # SIGNED ZERO (PLATFORM DEPENDENT) - OTHERWISE WRONG BEHAVIOUR
  if minn == 0.0:
    maxx = -0.0 
          
  if abs(minn) > abs(maxx):
    a = abs(minn)
  else:
    a = abs(maxx)

  vmin = -a 
  vmax = a 
  
  return vmin, vmax
def colorbar(mappable, ax, pos='right', size='3%', pad=0.2):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  divider = make_axes_locatable(ax)
  cax = divider.append_axes(pos, size, pad)
  cbar = ax.figure.colorbar(mappable, cax=cax) 
  plt.sca(ax)
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
