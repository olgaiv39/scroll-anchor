#!/usr/bin/env python3
"""Run the synthetic benchmark through the package CLI."""
from __future__ import annotations

import sys

from scroll_anchor.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
