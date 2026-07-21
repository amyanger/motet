"""Deprecated. Use `vexas serve` instead."""

import sys

from vexas.cli import main

if __name__ == "__main__":
    print("run_web.py is deprecated -- use: vexas serve --limit 25\n")
    sys.exit(main(["serve", *sys.argv[1:]]))
