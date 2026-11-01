from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantpulse_quality import audit_stock_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only QuantPulse OHLCV quality checks")
    parser.add_argument("csv", nargs="?", default="stock_market_data.csv", help="Path to the source OHLCV CSV")
    parser.add_argument("--output-dir", default="quantpulse_quality_output", help="Directory for audit outputs")
    args = parser.parse_args()

    source = Path(args.csv)
    if not source.is_file():
        parser.error(f"CSV not found: {source}")

    frame = pd.read_csv(source)
    result = audit_stock_data(frame, source_path=source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.checks.to_csv(output / "data_quality_checks.csv", index=False)
    payload = {"summary": result.summary, "fingerprint_sha256": result.fingerprint_sha256,
               "failed_checks": result.failed_count, "warning_checks": result.warning_count,
               "checks": result.checks.to_dict(orient="records")}
    (output / "data_quality_report.json").write_text(json.dumps(payload, indent=2, default=str))

    print(pd.DataFrame([result.summary]).to_string(index=False))
    print("\n", result.checks.to_string(index=False))
    print(f"\nSHA-256: {result.fingerprint_sha256 or 'unavailable'}")
    print(f"Failures: {result.failed_count}; Warnings: {result.warning_count}")
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
