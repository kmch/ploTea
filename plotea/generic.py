from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class FigFactory:
  subclasses = {}
  @classmethod
  def create(cls, id, *args, **kwargs):
    import p2d.mpl2d
    if id not in cls.subclasses:
      raise ValueError('Wrong id {}'.format(id))
    return cls.subclasses[id](*args, **kwargs)
  @classmethod
  def register_subclass(cls, id):
    def decorator(subclass):
      cls.subclasses[id] = subclass
      return subclass
    return decorator
class Fig(ABC):
  @abstractmethod
  def __init__(self, xsize, ysize):
    pass
  @abstractmethod
  def _add_all_axes(self):
    pass
  @abstractmethod
  def _add_axes_2d(self):
    pass
  @abstractmethod
  def save(self, fname):
    pass
