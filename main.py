from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from okx_report import analyze_csv, discover_csv_path, print_summary, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze an OKX unified account ledger CSV and generate a report.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Path to the OKX CSV export. Defaults to the only CSV file in the project root.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the full traceback when an error occurs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent

    try:
        csv_path = discover_csv_path(project_root, args.csv_path)
        result = analyze_csv(csv_path)
        output_paths = write_outputs(result, project_root / "reports")
        print_summary(result, output_paths)
        return 0
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
