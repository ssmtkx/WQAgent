"""Allow ``python -m wqagent <task>`` in addition to the ``wq`` console script."""

from __future__ import annotations

from wqagent.cli.app import main

if __name__ == "__main__":
    main()
