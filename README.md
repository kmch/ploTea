# :leaves: ploTea 

A framework for data-rich, interactive, publication-quality figures. Making the most of `matplotlib`, `plotly`, `pyvista`, `pygmt` and more!

## Features
- self-consistent 2D slices through volumes, surfaces and point-clouds provided by `arrau`,
  - intelligent visualisation of slicing lines,
  - multi-layer plots with isolines, shading and transparency,
- 3D rendering:
  - by `pyvista` (`vtk`-powered),
  - by `ipyvolume` (`OpenGL`-powered), 
- interactive display of associated metadata using `ipywidgets` and `plotly`,
- geographical coordinate-systems provided by `pygmt`.

## Installation
For the lastest stable version run:
```bash
conda install kmch::plotea
```

For the most up-to-date but potentially unstable version, run:
```bash
git clone git@github.com:kmch/ploTea.git
cd ploTea
pip install -e .
```

## Citing
If you use any part of this repository in your research, please cite the following paper:

-  Chrapkiewicz, Kajetan and Lipp, Alex and Barron, Leon Patrick and Barnes, Richard and Roberts, Gareth, **Apportioning sources of chemicals of emerging concern along an urban river with inverse modelling**. Science of the Total Environment. https://doi.org/10.1016/j.scitotenv.2024.172827
