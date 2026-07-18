"""
Custom-formatted log messages visible in Jupyter.

Notes
-----
Use it in every module as::

    from plotea.log import get_logger
    _log = get_logger(__name__)

    def foo():
        _log.info('...')

    class Foo:
        def foo(self):
            _log.info('...')

and switch it on once, from the notebook or script::

    import plotea
    plotea.set_log_level()

The call site is a bare module-level ``_log`` in both cases. Everything that
makes ``Foo.foo`` print as ``Foo.foo`` rather than ``foo`` happens in a Filter
on the handler, so no class has to hold a logger of its own.

"""
import logging
import sys

_FORMAT = '%(levelname)s:%(name)s:%(where)s: %(message)s'
_ROOT_NAME = 'plotea'


class _QualNameFilter(logging.Filter):
    """
    Set ``record.where`` to 'Class.method' for methods and 'function' for plain functions.

    Notes
    -----
    Attach this to the HANDLER, not to a logger. Filters do not propagate to
    child loggers, so a filter on the 'plotea' logger would never see a record
    from 'plotea.maps.carto'. On the handler it sees every record, which also
    guarantees ``record.where`` always exists so ``_FORMAT`` cannot raise on a
    record emitted by third-party code.

    The class name comes from ``code.co_qualname`` (python >= 3.11) on the
    caller's frame, located by matching pathname and function name against the
    record. Reading ``co_qualname`` rather than sniffing ``f_locals['self']``
    means ``@staticmethod`` resolves correctly, which the self-sniffing approach
    structurally cannot. The tradeoff: an inherited method reports the class that
    DEFINED it, not the runtime class -- which is what you want when the point of
    the prefix is to find the source.

    The frame walk only runs once a record already exists, i.e. after
    ``isEnabledFor``, so suppressed DEBUG calls cost nothing.

    Examples
    --------
    >>> handler.addFilter(_QualNameFilter())

    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Populate ``record.where``. Always returns True; this filter never drops records.

        Examples
        --------
        >>> _QualNameFilter().filter(record)
        True

        """
        where = record.funcName
        frame = sys._getframe(0)
        while frame is not None:
            code = frame.f_code
            if code.co_filename == record.pathname and code.co_name == record.funcName:
                where = getattr(code, 'co_qualname', record.funcName)
                break
            frame = frame.f_back
        record.where = where
        return True


class _StdoutHandler(logging.StreamHandler):
    """
    A StreamHandler that resolves ``sys.stdout`` at emit time rather than at construction.

    Notes
    -----
    ``logging.StreamHandler(stream=sys.stdout)`` binds the stream OBJECT once,
    when the handler is built. Anything that later swaps ``sys.stdout`` -- pytest's
    capture, ``contextlib.redirect_stdout``, some Jupyter kernels -- is then bypassed
    and the message escapes to the original stream. Resolving lazily through a
    property fixes that and costs nothing.

    stdout rather than stderr, so notebook output appears as ordinary cell output
    instead of a red-boxed stream.

    Examples
    --------
    >>> handler = _StdoutHandler()

    """

    @property
    def stream(self):
        """
        The current ``sys.stdout``, resolved on every access.

        Examples
        --------
        >>> _StdoutHandler().stream is sys.stdout
        True

        """
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        """
        Ignore writes. ``StreamHandler.__init__`` assigns to ``stream``; we resolve it lazily instead.

        Examples
        --------
        >>> h = _StdoutHandler()   # StreamHandler.__init__ assigns here, harmlessly

        """


def get_logger(name: str) -> logging.Logger:
    """
    Return the logger for a module. Use as ``_log = get_logger(__name__)`` at module level.

    Parameters
    ----------
    name : str
        Always ``__name__``. Gives 'plotea.maps.base' and friends, so log lines
        say which module spoke and ``set_log_level`` can configure them as a group.

    Returns
    -------
    logging.Logger

    Examples
    --------
    >>> _log = get_logger(__name__)
    >>> _log.info('assuming coordinates are lon/lat')

    """
    return logging.getLogger(name)


def set_log_level(level: int = logging.INFO) -> logging.Logger:
    """
    Switch plotea's logging on. Call once, from a notebook or script.

    Parameters
    ----------
    level : int
        ``logging.DEBUG``, ``logging.INFO`` (default), ``logging.WARNING``, ...

    Returns
    -------
    logging.Logger
        The 'plotea' logger, already configured.

    Notes
    -----
    Attaches one handler to the 'plotea' logger and sets ``propagate = False``,
    rather than calling ``logging.basicConfig(force=True)`` on the root logger.
    Two reasons, both deliberate: ``force=True`` rips out any handler the host
    application installed, and leaving propagation on makes Jupyter's own root
    handler print every message a second time.

    Idempotent -- calling it twice replaces plotea's handler instead of stacking
    a second one, so repeated calls in a notebook cannot cause double printing.

    Examples
    --------
    >>> import plotea
    >>> plotea.set_log_level()
    >>> fig, ax = plotea.BaseMap().plot()
    INFO:plotea.maps.base:BaseMap.plot: whole world, EqualEarth, resolution 50m

    """
    log = logging.getLogger(_ROOT_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_QualNameFilter())
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False
    return log
