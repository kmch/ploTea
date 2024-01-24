# :leaves: ploTea 

A framework for data-rich, interactive, publication-quality figures. Making the most of
`matplotlib`, `plotly`, `pyvista`, `pygmt` and more!

## Installation
`conda install kmch::plotea`

## Features
- self-consistent 2D slices through volumes, surfaces and point-clouds provided by `arrau`,
  - intelligent visualisation of slicing lines,
  - multi-layer plots with isolines, shading and transparency,
- 3D rendering:
  - by `pyvista` (`vtk`-powered),
  - by `ipyvolume` (`OpenGL`-powered), 
- interactive display of associated metadata using `ipywidgets` and `plotly`,
- geographical coordinate-systems provided by `pygmt`.
