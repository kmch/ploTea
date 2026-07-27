"""
Tests for plotea's logging: Class.method prefixes, Jupyter/pytest visibility, no double print.

"""
import io
import logging
from contextlib import redirect_stdout

from plotea.log import get_logger, init_logging

_log = get_logger('plotea.test_log')


def _function_logs():
    """
    Emit one INFO line from a plain function.

    Examples
    --------
    >>> _function_logs()

    """
    _log.info('from a function')


class _Thing:
    """
    A class whose method, classmethod and staticmethod all log.

    Examples
    --------
    >>> _Thing().method()

    """

    def method(self):
        """
        Log from an instance method.

        Examples
        --------
        >>> _Thing().method()

        """
        _log.info('from a method')

    @classmethod
    def klass(cls):
        """
        Log from a classmethod.

        Examples
        --------
        >>> _Thing.klass()

        """
        _log.info('from a classmethod')

    @staticmethod
    def stat():
        """
        Log from a staticmethod.

        Examples
        --------
        >>> _Thing.stat()

        """
        _log.info('from a staticmethod')


def _capture(action):
    """
    Configure logging, run ``action`` under a stdout redirect, return captured text.

    Examples
    --------
    >>> 'method' in _capture(_Thing().method)

    """
    init_logging(logging.INFO)
    buf = io.StringIO()
    with redirect_stdout(buf):
        action()
    return buf.getvalue()


def test_function_prefix_is_bare_name():
    """
    A plain function logs its bare name, no class prefix.

    Examples
    --------
    >>> test_function_prefix_is_bare_name()

    """
    out = _capture(_function_logs)
    assert '_function_logs: from a function' in out


def test_method_prefix_has_class_and_method():
    """
    An instance method logs 'Class.method'.

    Examples
    --------
    >>> test_method_prefix_has_class_and_method()

    """
    out = _capture(_Thing().method)
    assert '_Thing.method: from a method' in out


def test_classmethod_and_staticmethod_prefixes():
    """
    classmethod and staticmethod both log 'Class.method' -- the staticmethod case is why
    co_qualname is used instead of sniffing ``self``.

    Examples
    --------
    >>> test_classmethod_and_staticmethod_prefixes()

    """
    assert '_Thing.klass: from a classmethod' in _capture(_Thing.klass)
    assert '_Thing.stat: from a staticmethod' in _capture(_Thing.stat)


def test_lazy_stdout_capture():
    """
    The handler resolves stdout at emit time, so a redirect installed AFTER configuration
    still captures. An eager StreamHandler(stream=sys.stdout) would escape it.

    Examples
    --------
    >>> test_lazy_stdout_capture()

    """
    init_logging(logging.INFO)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _log.info('captured lazily')
    assert 'captured lazily' in buf.getvalue()


def test_no_double_print_with_root_handler():
    """
    A pre-installed root handler must not print plotea's messages a second time.

    Examples
    --------
    >>> test_no_double_print_with_root_handler()

    """
    root = logging.getLogger()
    sentinel = io.StringIO()
    root_handler = logging.StreamHandler(sentinel)
    root.addHandler(root_handler)
    try:
        init_logging(logging.INFO)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _log.info('single')
        assert buf.getvalue().count('single') == 1
        assert 'single' not in sentinel.getvalue()
    finally:
        root.removeHandler(root_handler)


def test_repeated_configuration_does_not_stack_handlers():
    """
    Calling ``init_logging`` twice replaces the handler rather than adding a second.

    Examples
    --------
    >>> test_repeated_configuration_does_not_stack_handlers()

    """
    init_logging(logging.INFO)
    init_logging(logging.INFO)
    assert len(logging.getLogger('plotea').handlers) == 1
