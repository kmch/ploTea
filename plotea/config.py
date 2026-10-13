"""
Where things are: derived from where the package sits, never written down.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    """
    The directories plotea reads from.

    A convenience for call sites -- a notebook, a test, a docs page -- and not a dependency
    the library reaches into: every function takes explicit paths and this only supplies a
    default when one is not given. The root is worked out from the installed package, so
    nothing carries anyone's home directory and a checkout somewhere else gets its own paths
    for free. Frozen, so a path cannot move under something that has already read half of it.

    Parameters
    ----------
    root : Path
        The repository root.

    Examples
    --------
    >>> PATH.datasets / 'sardinia_elv.tif'

    """

    root : Path

    @property
    def datasets(self) -> Path:
        """
        Small rasters shipped for the notebooks, the tests and the docs.

        Outside the installed package on purpose: ``pyproject.toml`` packages ``plotea*``
        only, so these live in the repository and never go into a wheel. Somebody who
        installed plotea from conda has the library, not the demo data, which is the right
        way round for a plotting package.

        Examples
        --------
        >>> PATH.datasets

        """
        return self.root / 'datasets'

    @property
    def styles(self) -> Path:
        """
        Matplotlib style sheets, including the one the journal rules are encoded in.

        Examples
        --------
        >>> plt.style.use(PATH.styles / 'plotea.mplstyle')

        """
        return self.root / 'styles'

    @classmethod
    def from_package(cls) -> 'Paths':
        """
        The paths belonging to this checkout of the package.

        ``plotea/config.py`` sits one level below the root, so the root is its parent's
        parent. An installed-not-editable copy has no repository above it, in which case
        ``datasets`` simply will not exist and a caller gets a clear missing-file error
        rather than a path into site-packages.

        Examples
        --------
        >>> Paths.from_package()

        """
        return cls(root=Path(__file__).resolve().parent.parent)

#: The paths for this checkout, for call sites that do not want to build their own.
PATH = Paths.from_package()
