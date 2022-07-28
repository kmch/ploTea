import ipyvolume as ipv

class Ipv:
  @classmethod
  def volshow(cls, arr, **kwargs):
    """
    kwargs:   
        azimuth (float) –   rotation around the axis pointing up in degrees
        elevation (float) – rotation where +90 means ‘up’, -90 means ‘down’,
                            in degrees
        distance (float) – radial distance from the center to the camera.
    """
    ipv.figure(width=kwargs.get('width', 400), 
               height=kwargs.get('height',500))
    ipv.volshow(arr, 
      controls=kwargs.get('controls', False), # show/hide sliders
      memorder='F', # this is crucial, otherwise x and z axes are swapped 
      extent=kwargs.get('extent', None),
      level=[1,0,0], # 'tuned' manually but its meaning is obscure, see:
      # the official example Visualizating-a-scan-of-a-male-head
      level_width=kwargs.get('level_width', 1), # if one, entire grid cell is filled
      max_opacity=kwargs.get('max_opacity', 1), 
      opacity=kwargs.get('opacity', [1,1,1]),
      downscale=kwargs.get('downscale', 1), # decimate big arrays
      # changing the below seems to have little effect:
      stereo=False, #stereo view for virtual reality 
      lighting=True, # no diff
      # lighting params
      ambient_coefficient=0.5, diffuse_coefficient=0.8, 
      specular_coefficient=0.5, specular_exponent=5
    )   
    ipv.view(
      azimuth=kwargs.get('azimuth', None),
      elevation=kwargs.get('elevation', None),
      distance=kwargs.get('distance', None)
    )
    xlabel = kwargs.get('xlabel', 'x, m')
    ylabel = kwargs.get('ylabel', 'x, m')
    zlabel = kwargs.get('zlabel', 'z, m')
    ipv.xyzlabel(xlabel, ylabel, zlabel)
    return ipv.show()