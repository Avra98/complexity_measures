
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    here = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(here, "grokking"))
    from grokking.run_modular_addition import main as run  # type: ignore

    # Wrapper CLI: we forward args to grokking/run_modular_addition.py, but expose
    # --device here for convenience.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0, cuda:1")
    args, rest = parser.parse_known_args()

    # If provided here, inject/override --device for the downstream script.
    if args.device is not None:
        dev = args.device.strip()
        # Allow shorthand: "--device 7" meaning cuda:7
        if dev.isdigit():
            dev = f"cuda:{dev}"
        # Remove any existing --device ... from rest so ours wins.
        cleaned: list[str] = []
        skip_next = False
        for tok in rest:
            if skip_next:
                skip_next = False
                continue
            if tok == "--device":
                skip_next = True
                continue
            cleaned.append(tok)
        rest = cleaned + ["--device", dev]

    sys.argv = [sys.argv[0]] + rest
    run()


if __name__ == "__main__":
    main()


