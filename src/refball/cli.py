"""Convenience dispatcher exposed as the ``refball`` console script.

Each stage is *also* runnable directly (``python -m refball.data.pull ...``); this just
gives a single entry point. Stage modules are imported lazily so that, e.g., running the
data pull does not import PyMC.
"""

from __future__ import annotations

import argparse
import sys

_COMMANDS = {
    "pull": ("refball.data.pull", "main"),
    "build-table": ("refball.features.build_table", "main"),
    "eda": ("refball.features.eda", "main"),
    "fit-stage1": ("refball.models.fit_stage1", "main"),
    "fit-stage2": ("refball.models.fit_stage2", "main"),
    "mediation": ("refball.models.mediation", "main"),
    "robustness": ("refball.models.robustness", "main"),
    "synth": ("refball.data.synthetic", "main"),
    "l2m": ("refball.data.l2m", "main"),
    "l2m-model": ("refball.models.l2m_model", "main"),
    "within-series": ("refball.models.within_series", "main"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="refball", description="Ref Ball pipeline dispatcher")
    parser.add_argument("command", choices=sorted(_COMMANDS), help="stage to run")
    ns, rest = parser.parse_known_args(argv[:1])
    import importlib

    mod_name, fn_name = _COMMANDS[ns.command]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return int(fn(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
