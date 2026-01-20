"""Batch prewetting comparison helper.

Loads the wet/dry folders that were previously analysed in
``MA_Prewetting_Compare_V2.ipynb`` with ``solarSimulator``. The script
collects Voc, Isc, FF, and PCE for every measurement, computes the average and
standard-error per metric/group, and stores the result as a CSV inside
``result_dir``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from plot_module.solsim_analyzer import solarSimulator

DATA_ROOT = Path(
    ""  # Enter your file path/folder path in this place
    "Pervoskite Space(Master)/Data/SolSim/MA_Data/With_Without_Prewetting/Data_Used"
)

DEFAULT_GROUPS: Mapping[str, Path] = {
    "With Prewetting": DATA_ROOT / "With_Prewetting",
    "Without Prewetting": DATA_ROOT / "Without_Prewetting",
}
DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parent / "Result_Average" / "MA_Prewetting_Compare_V2"
)
METRICS = ("Voc", "Isc", "FF", "PCE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate wet/dry prewetting metrics and write averages/errors."
    )
    parser.add_argument(
        "--with-folder",
        default=str(DEFAULT_GROUPS["With Prewetting"]),
        help="Directory that contains the measurements with prewetting.",
    )
    parser.add_argument(
        "--without-folder",
        default=str(DEFAULT_GROUPS["Without Prewetting"]),
        help="Directory that contains the measurements without prewetting.",
    )
    parser.add_argument(
        "--with-ref-current",
        type=float,
        default=32.0,
        help="Reference current for the with-prewetting analyzer.",
    )
    parser.add_argument(
        "--without-ref-current",
        type=float,
        default=32.0,
        help="Reference current for the without-prewetting analyzer.",
    )
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Destination directory for the CSV export.",
    )
    parser.add_argument(
        "--filename",
        default="prewetting_metric_summary.csv",
        help="Filename of the CSV that stores the averages/errors.",
    )
    return parser.parse_args()


def load_analyzer(folder: Path, ref_current: float) -> solarSimulator:
    if not folder.is_dir():
        raise FileNotFoundError(f"Measurement folder missing: {folder}")
    analyzer = solarSimulator(folderPath=str(folder), refCurrent=ref_current)
    analyzer.loadFolderData()
    return analyzer


def build_metric_records(
    analyzers: Mapping[str, solarSimulator],
) -> list[dict[str, str | float]]:
    records: list[dict[str, str | float]] = []
    for label, analyzer in analyzers.items():
        for metric in METRICS:
            values = getattr(analyzer, metric, None) or []
            for value in values:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                records.append({"Group": label, "Metric": metric, "Value": numeric})
    if not records:
        raise RuntimeError("No valid metric values found; check the folders.")
    return records


def compute_summary(records: list[dict[str, str | float]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["Metric", "Group"])["Value"]
        .agg(sample_count="count", average="mean", std_dev="std")
        .reset_index()
    )
    summary["error"] = summary["std_dev"] / summary["sample_count"].pow(0.5)
    summary = summary.sort_values(["Metric", "Group"]).reset_index(drop=True)
    return summary


def ensure_result_dir(path: str | Path) -> Path:
    result_dir = Path(path).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def main() -> None:
    args = parse_args()
    group_paths = {
        "With Prewetting": Path(args.with_folder).expanduser(),
        "Without Prewetting": Path(args.without_folder).expanduser(),
    }
    analyzers = {
        label: load_analyzer(folder, ref_current)
        for label, folder, ref_current in (
            ("With Prewetting", group_paths["With Prewetting"], args.with_ref_current),
            (
                "Without Prewetting",
                group_paths["Without Prewetting"],
                args.without_ref_current,
            ),
        )
    }
    records = build_metric_records(analyzers)
    summary = compute_summary(records)

    result_dir = ensure_result_dir(args.result_dir)
    output_path = result_dir / args.filename
    summary.to_csv(output_path, index=False)
    print(f"Wrote metric summary to {output_path}")


if __name__ == "__main__":
    main()
