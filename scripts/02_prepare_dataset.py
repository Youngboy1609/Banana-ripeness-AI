from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.data.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    prepare_dataset(config_path, ROOT)


if __name__ == "__main__":
    main()
