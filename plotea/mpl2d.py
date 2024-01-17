"""
Wrapper around matplotlib 2D plotting functions. 
"""
from abc import ABC, abstractmethod
import cmocean
from autologging import logged, traced
import matplotlib.colors as mcolors # for cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource #, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# from plotea.generic import FigFactory, Fig

class AxesFormatter:
  def __init__(self, ax=None, extent=None, unit=None):
    self.ax = ax
    self.extent = extent
    self.unit = unit
  def _adjust(self, ax, labels=True, ticklabels=True, ticks=True, \
    rotate_ylabels=False, **kwargs):
    if not labels:
      ax.set_xlabel(None)
      ax.set_ylabel(None)
    if not ticklabels:
      ax.set_xticklabels([])
      ax.set_yticklabels([])
    if not ticks:
      ax.set_xticks([])
      ax.set_yticks([])
    if rotate_ylabels:
      plt.yticks(rotation=90, va='center')
  def _convert_ticks_m2km(self, ax):
    ax = divide_ticklabels_values(ax, axis='both', factor=1000, round_to_int=1)
    return ax            
  def _set_xylabels(self):
    plt.gca().set_xlabel("Easting (km)")
    plt.gca().set_ylabel("Northing (km)")
  def _set_lims(self, ax=None):
    ax = get_ax(ax)
    ax.set_xlim(self.extent[:2])
    ax.set_ylim(self.extent[2:])
    return ax




def plot_boxplot(df, c='lightgrey',  ax=None, rot=0, vert=0, alpha=1.,\
  connect_means=False, grid=None, aspect='auto', outliers=True, lw=1,
  norm='log', **kwargs):

  ax = get_ax(ax)    
  meancolor = 'r'
  ce = 'k'
  outlier_edgecolor = 'k' if outliers else 'none'
  df.boxplot(ax=ax, rot=rot, showmeans=1, vert=vert, widths=.5, \
      patch_artist=True, # to fill in boxes
      boxprops={'facecolor': c, 'edgecolor': ce, 'alpha': alpha, 'lw': lw},
      capprops={'color': ce, 'lw': lw}, # ends of whiskers
      flierprops={'markerfacecolor': 'none', 'markeredgecolor': outlier_edgecolor, 'marker': 'o'},
      meanprops={'marker': '^', 'markerfacecolor': 'w', 'markeredgecolor': ce, \
        'markersize': 10, 'markeredgewidth': lw},
      medianprops={'color': 'r', 'lw': lw, "zorder":100},
      whiskerprops={'color': ce, 'lw': lw}, **kwargs,
  )
  if norm == 'log':
    if vert:
      ax.set_yscale('log')
    else:
      ax.set_xscale('log')
  if connect_means:
    means = df.mean()
    x, y = range(1, len(means) + 1), means
    if not vert:
      x, y = y, x
    ax.plot(x, y, c=meancolor)
  ax.set_aspect(aspect)
  ax.grid(grid)
  return ax
def plot_crossplot(x, y,  ax=None, logx=True, logy=True, pad=1, lim=None,
  cbar=True, annotations=None, c=None, edgecolor='k', s=100, alpha=1, linecolor='k', linestyle='--'):
  """
  Cross-plot examples: 
  - predicted vs. observed data, 
  - misfit vs. roughness (data-residual norm vs. model norm),
  - ...

  """
  ax = get_ax(ax)
  
  x, y = np.array(x), np.array(y)
  x = np.log10(x) if logx else x
  y = np.log10(y) if logy else y

  sc = ax.scatter(x, y, color=c, edgecolor=edgecolor, s=s, alpha=alpha, zorder=2)
  
  if annotations is not None:
    for xa, ya, ann in zip(x, y, annotations):
      ax.annotate(ann, (xa, ya), c='k')
  if cbar:
    plt.colorbar(sc)

  x1 = np.min([min(x), min(y)])
  x2 = np.max([max(x), max(y)])
  xlim = (x1-pad, x2+pad) if lim is None else lim
  ylim = xlim

  x = np.linspace(*xlim, 100)
  y = x
  ax.plot(x, y, c=linecolor, linestyle=linestyle, zorder=1)

  return ax


# utils
def plot_arrow(xd, yd, angle, width, pad, ax=None, cvalue=None, edgecolor='k', \
  cmap='Reds', vmin=0, vmax=5):
  ax = get_ax(ax)

  cmap = plt.get_cmap(cmap)
  value_to_map = 3.7
  norm = plt.Normalize(vmin, vmax)
  normalized_value = norm(cvalue)
  color = cmap(normalized_value)

  head_length = width * 2
  head_width = 2 * width
  length = 4 * width
  angle = -angle * np.pi / 180
  x = xd - (length + head_length) * (1 + pad) * np.sin(angle)
  y = yd - (length + head_length) * (1 + pad) * np.cos(angle)
  dx = length * np.sin(angle)
  dy = length * np.cos(angle)




  ax.arrow(x=x, y=y, dx=dx, dy=dy, width=width, facecolor=color, edgecolor=edgecolor,
            head_length=head_length, head_width=head_width)
  return ax
def plot_arrow_from_a_to_b(a, b, ax=None, c='turquoise', edgecolor=None, width=1,\
   head_length=None, head_width=None, alpha=0.5, zorder=None):
  ax = get_ax(ax)
  head_width = 2 * width if head_width is None else head_width 
  head_length = 2 * width if head_length is None else head_length
  
  x1, y1 = a
  x2, y2 = b
  dx = x2 - x1 
  dy = y2 - y1
  assert head_length < np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  angle = np.arctan2((y2 - y1), (x2 - x1))
  dx = (x2 - x1) - head_length * np.cos(angle)
  dy = (y2 - y1) - head_length * np.sin(angle)
  ax.arrow(x=x1, y=y1, dx=dx, dy=dy, facecolor=c, edgecolor=edgecolor,\
    width=width, head_length=head_length, head_width=head_width, alpha=alpha, 
    zorder=zorder)
  return ax





#FIXME DEL?
def boxplot(df, c, rot=0, vert=0, ax=None, alpha=1., connect_means=True, \
  grid=True, aspect='auto', colorscheme=1, **kwargs):
  if ax is None:
    fig, ax = plt.subplots()      
  if colorscheme == 1:
    meancolor = 'g'
    df.boxplot(ax=ax, rot=rot, showmeans=1, vert=vert, \
      patch_artist=True, # to fill in boxes
      boxprops={'facecolor': 'none', 'edgecolor': 'k', 'alpha': alpha},
      # capprops={'color': c}, # ends of whiskers
      # flierprops={'markerfacecolor': 'none', 'markeredgecolor': c, 'marker': 'o'},
      meanprops={'markerfacecolor': meancolor, 'markeredgecolor': 'none'},
      medianprops={'color': 'r'},
      whiskerprops={'color': 'k'}, **kwargs,
    )
  elif colorscheme == 2:
    meancolor = c
    df.boxplot(ax=ax, rot=rot, showmeans=1, vert=vert, \
      patch_artist=True, # to fill in boxes
      boxprops={'facecolor': 'none', 'edgecolor': c, 'alpha': alpha},
      capprops={'color': c}, # ends of whiskers
      flierprops={'markerfacecolor': 'none', 'markeredgecolor': c, 'marker': 'o'},
      meanprops={'markerfacecolor': meancolor, 'markeredgecolor': 'none'},
      medianprops={'color': c},
      whiskerprops={'color': c}, **kwargs,
    )
  else:
    meancolor = 'k'
    df.boxplot(ax=ax, rot=rot, showmeans=1, vert=vert, \
      patch_artist=True, # to fill in boxes
      boxprops={'facecolor': c, 'edgecolor': 'k', 'alpha': alpha},
      capprops={'color': 'k'}, # ends of whiskers
      flierprops={'markerfacecolor': 'none', 'markeredgecolor': c, 'marker': 'o'},
      meanprops={'markerfacecolor': meancolor, 'markeredgecolor': 'none'},
      medianprops={'color': 'k'},
      whiskerprops={'color': 'k'}, **kwargs,
    )
  if connect_means:
    means = df.mean()
    x, y = range(1, len(means) + 1), means
    if not vert:
      x, y = y, x
    ax.plot(x, y, c=meancolor)
  ax.set_aspect(aspect)
  return ax





# @FigFactory.register_subclass('mpl2d')
# class FigMpl(Fig):
#   """
#   Matplotlib figure with 2d axes.

#   Notes
#   -----
#   This object-oriented interface serves to 
#   create finely-tuned, complex figures.

#   For quick plots on an existing figure,
#   just use the standalone plot() function.
#   """
#   def __init__(self, xsize=6, ysize=6, nrows=1, ncols=1,\
#     projection='2d', style='default'):
#     """
#     """
#     self._set_size(xsize, ysize)
#     self._set_style(style)
    
#     self.fig = plt.figure(figsize=self.figsize)
#     self._init_all_axes(nrows, ncols)
#     # self._create_all_axes()
#     # self._set_current_axes()
#     self.cax = plt.gca()
#   def plot_image(self, image):
#     return self.cax.imshow(image)  
#   def plot_line(self, y):
#     return self.cax.plot(y)
#   def save(self, *args, **kwargs):
#     """
#     Format is deduced from the fname.

#     """
#     plt.savefig(*args, **kwargs)
#   # -----------------------------------------------------------------------------
#   def _add_all_axes(self, nrows, ncols, projs='2d'):
#     """
#     """
#     from matplotlib.gridspec import GridSpec
#     self.gs = GridSpec(nrows, ncols)
    
#     for x in range(nrows):
#       for y in range(ncols):
#         self.axes[x][y] = self._add_axes2d(x, y, self.projection[x,y])
#   def _add_axes_2d(self, x, y, projection):
#     self.axes[x][y] = self.fig.add_subplot(self.gs[x,y],\
#        projection=projection)
#   def _add_layout(self):
#     from matplotlib.gridspec import GridSpec
#     self.gs = GridSpec(nrows, ncols)
#   def _create_axes(self, nrows, ncols, projection='2d'):
#     """
#     You can later set_height_rat.shell etc.
    
#     # cax = fig.add_subplot(111) # 111 needed, otherwise
#     # ... returns None

#     """
#     from matplotlib.gridspec import GridSpec
#     from mpl_toolkits.mplot3d import Axes3D
    
#     self.gs = GridSpec(nrows, ncols)
#     self._set_axes_projections(projection, nrows, ncols)

#     self.axes = [] # ARRAY CAN'T STORE SUBPLOTS APPARENTLY
#     for x in range(nrows):
#       tmp = []
#       for y in range(ncols):
#         tmp.append(\
#           self.fig.add_subplot(self.gs[x,y], projection=self.projection[x,y]))
#       self.axes.append(tmp)
#   def _init_all_axes(self, nrows, ncols):
#     self.axes = np.zeros((nrows, ncols)).tolist()
#     # self._set_axes_projections(projection, nrows, ncols)  
#   def _set_axes_projections(self, projection, nrows, ncols):
#     """
#     """
#     if isinstance(projection, str):
#       projection = np.full((nrows, ncols), projection)
#     elif isinstance(projection, np.arrauray):
#       assert projection.shape == (nrows, ncols)
#       projection = projection
#     else:
#       raise TypeError('type(projection)', type(projection))
#     self.projection = np.where(projection=='2d', None, projection)
#   def _set_current_axes(self, x=0, y=0):
#     self.cax = self.axes[x][y]
#   def _set_size(self, xsize, ysize):
#     self.figsize = (xsize, ysize)
#   def _set_style(self, style):
#     plt.style.use(['default', 'ggplot', style])
# # -------------------------------------------------------------------------------
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
    self._invert_vertical_axis(**dict(kwargs, ax=self.ax))
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

class Imshow(PlotterMpl):
  def _add_colorbar(self, mappable):
    divider = make_axes_locatable(self.ax)
    cax = divider.append_axes(**self.kwargs_divider)
    cbar = self.ax.figure.colorbar(mappable, cax=cax, **self.kwargs_cbar) 
    plt.yticks(rotation=90, va='center')
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
    self.mappable = mappable
    shaded = LightSource(**self.kwargs_light_source).shade(arr.T, **self.kwargs_shade)
    im = self.ax.imshow(shaded, **self.kwargs_imshow) # actual plot
    if self.kwargs_misc['cbar']:
      self._add_colorbar(mappable)
    return self.ax
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
@traced
def plot_rect(x1, x2, y1, y2, ax=None, ls='--', c='r', \
  lw=2, alpha=.6, label=None, zorder=None, aspect='equal'):
  """
  Plot a 2D box (rectangle) given its 2 vertices
  along the diagonal.
  
  Parameters
  ----------
  x1 : float 
    Min. value of X-coord.
  x2 : float 
    Max. value of X-coord.
  y1 : float 
    Min. value of Y-coord.
  y2 : float 
    Max. value of Y-coord.
  Returns
  -------
  None
  
  """
  ax = get_ax(ax)
  args = [ls]
  kwargs = dict(c=c, lw=lw, alpha=alpha, zorder=zorder)
  ax.plot([x1, x2], [y1, y1], *args, **kwargs, label=label)
  ax.plot([x1, x2], [y2, y2], *args, **kwargs)
  ax.plot([x1, x1], [y1, y2], *args, **kwargs)
  ax.plot([x2, x2], [y1, y2], *args, **kwargs)    
  ax.set_aspect(aspect)
  return ax


# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------
def aspeqt(ax=None, **kwargs):
  if ax is None:
    ax = plt.gca()
  ax.set_aspect('equal')
def autect(ax=None, **kwargs):
  if ax is None:
    ax = plt.gca()
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
def colorbar_flexible(im, ax, cbar_label): # new take, more flexible
  cb = plt.colorbar(im, ax=ax)
  cb.set_label(cbar_label)
def colorbar(mappable, ax, pos='right', size='3%', pad=0.2, label=None):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  divider = make_axes_locatable(ax)
  cax = divider.append_axes(pos, size, pad)
  cbar = ax.figure.colorbar(mappable, cax=cax, label=label)
  plt.sca(ax)
def colors(n, cmap='rainbow', **kwargs): # move to generic
  """
  Create an iterator for rainbow colors.
  
  Parameters
  ----------
  n : int 
    Number of colors in the spectrum.
  
  
  Returns
  -------
  colors : iterator
    c = next(colors)
  
  Notes
  -----
  Usage plot(..., c=next(colors))
  
  """
  from matplotlib.cm import get_cmap
  
  if isinstance(n, list):
    n = len(n)

  cmap = get_cmap(cmap)
  cols = iter(cmap(np.linspace(0, 1, n)))
  
  return cols
def divide_ticklabels_values(ax, axis='both', factor=1000, round_to_int=True):
  if axis == 'x' or axis == 'both':
    ticks_old = ax.get_xticks()
    ticks_new = ticks_old / factor
    ticks_new = [int(i) for i in ticks_new] if round_to_int else ticks_new
    ax.set_xticks(ticks_old)
    ax.set_xticklabels(ticks_new)
  if axis == 'y' or axis == 'both':
    ticks_old = ax.get_yticks()
    ticks_new = ticks_old / factor
    ticks_new = [int(i) for i in ticks_new] if round_to_int else ticks_new
    ax.set_yticks(ticks_old)
    ax.set_yticklabels(ticks_new)
  return ax
def convert_to_ax_coords(v, axis, ax):
  if axis == 'x':
    vmin, vmax = ax.get_xlim()
  elif axis == 'y':
    vmin, vmax = ax.get_ylim()
  else:
    raise ValueError('Unknown axis: %s' % axis)
  return (v - vmin) / (vmax - vmin)
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
def flipy(ax=None, **kwargs):
  if ax is None:
    ax = plt.gca()
  ax.invert_yaxis()
def get_ax(ax=None, figsize=None):
  if ax is None:
    _, ax = plt.subplots(figsize=None) 
  return ax
def remove_every_nth_tick_label(n: int, ax=None, axis='x', start=0,\
  remove_ticks=True):
  ax = get_ax(ax)
  get_ticks = plt.xticks if axis == 'x' else plt.yticks
  set_ticks = ax.set_xticks if axis == 'x' else ax.set_yticks
  # Get the current tick positions and labels
  ticks, ticklabels = get_ticks()
  ticks, ticklabels = ticks[start:], ticklabels[start:]
  # print(ticks)
  # print(ticklabels)
  # Remove every nth tick label
  if remove_ticks:
    ticks = ticks[::n]
    new_ticklabels = ticklabels[::n]
  else:
    new_ticklabels = [label if i % n == 0 else '' for i, label in enumerate(ticklabels)]
  # Set the modified tick labels
  set_ticks(ticks, new_ticklabels)
  return ax
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
# FIXME
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
