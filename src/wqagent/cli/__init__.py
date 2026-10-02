"""Command line interface for WQ Agent.

The Typer application lives in :mod:`wqagent.cli.app`. This package deliberately
re-exports nothing, so ``wqagent.cli.app`` keeps referring to the module and
callers (and tests) can patch it::

    from wqagent.cli.app import app, main
"""

__all__: list[str] = []
