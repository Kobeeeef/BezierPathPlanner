from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    _bootstrap_src_path()
    from path_planner.interactive_visualizer import launch_interactive_visualizer

    return launch_interactive_visualizer()


if __name__ == "__main__":
    raise SystemExit(main())
