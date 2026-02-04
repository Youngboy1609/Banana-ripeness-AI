from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.data.download import list_available_datasets


def main() -> None:
    external_root = ROOT / "data" / "external"
    datasets = list_available_datasets(external_root)
    if not datasets:
        print(f"[download] No datasets found under {external_root}")
        print("[download] Place datasets there and re-run.")
        return

    print("[download] Datasets already present:")
    for path in datasets:
        print(f"- {path}")


if __name__ == "__main__":
    main()
