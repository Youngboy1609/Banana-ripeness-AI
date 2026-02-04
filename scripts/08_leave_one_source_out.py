from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "source"


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-source-out training and evaluation")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--metadata", default="data/processed/metadata.csv", help="Metadata CSV path")
    parser.add_argument(
        "--output-dir",
        default="reports/leave_one_source_out",
        help="Base output directory for per-source results",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated list of sources to run (default: all)",
    )
    args = parser.parse_args()

    train_cfg = Path(args.config)
    if not train_cfg.is_absolute():
        train_cfg = ROOT / train_cfg

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path)
    sources = sorted(meta["source"].dropna().unique().tolist())

    if args.sources.strip():
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        sources = [s for s in sources if s in wanted]

    if not sources:
        print("[loo] No sources found to run.")
        return

    for source in sources:
        slug = slugify(source)
        print(f"[loo] Running holdout source: {source}")

        loo_meta = meta.copy()
        holdout_mask = loo_meta["source"] == source
        loo_meta.loc[holdout_mask, "split"] = "test"
        non_holdout_test = (~holdout_mask) & (loo_meta["split"] == "test")
        if non_holdout_test.any():
            moved_count = int(non_holdout_test.sum())
            loo_meta.loc[non_holdout_test, "split"] = "train"
            print(f"[loo] Moved {moved_count} non-holdout test samples to train for pure LOO.")

        loo_dir = ROOT / "data" / "processed" / "loo"
        loo_dir.mkdir(parents=True, exist_ok=True)
        loo_meta_path = loo_dir / f"metadata_{slug}.csv"
        loo_meta.to_csv(loo_meta_path, index=False)

        run_output = output_dir / slug
        run_output.mkdir(parents=True, exist_ok=True)

        cmd_train = [
            sys.executable,
            str(ROOT / "scripts" / "03_train.py"),
            "--config",
            str(train_cfg),
            "--metadata",
            str(loo_meta_path),
        ]
        subprocess.run(cmd_train, check=True)

        cmd_eval = [
            sys.executable,
            str(ROOT / "scripts" / "04_evaluate.py"),
            "--config",
            str(train_cfg),
            "--metadata",
            str(loo_meta_path),
            "--checkpoint",
            str(ROOT / "models" / "checkpoints" / "best.pth"),
            "--output-dir",
            str(run_output),
        ]
        subprocess.run(cmd_eval, check=True)

        train_log = ROOT / "reports" / "metrics" / "train_log.csv"
        if train_log.exists():
            metrics_dir = run_output / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(train_log, metrics_dir / "train_log.csv")

        checkpoint_dir = run_output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ROOT / "models" / "checkpoints" / "best.pth"
        last_ckpt = ROOT / "models" / "checkpoints" / "last.pth"
        if best_ckpt.exists():
            shutil.copy2(best_ckpt, checkpoint_dir / "best.pth")
        if last_ckpt.exists():
            shutil.copy2(last_ckpt, checkpoint_dir / "last.pth")

        print(f"[loo] Completed: {source} -> {run_output}")


if __name__ == "__main__":
    main()
