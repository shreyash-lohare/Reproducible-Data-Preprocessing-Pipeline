import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, load_config, load_raw_dataset, save_json, setup_logger


def _check_schema(df: pd.DataFrame, expected_columns: list) -> dict:
	actual_cols = list(df.columns)
	missing = [c for c in expected_columns if c not in actual_cols]
	extra = [c for c in actual_cols if c not in expected_columns]
	return {
		"status": "pass" if not missing and not extra else "fail",
		"missing_columns": missing,
		"extra_columns": extra,
	}


def _missing_values(df: pd.DataFrame) -> dict:
	return df.isna().sum().to_dict()


def _label_distribution(df: pd.DataFrame, target: str) -> dict:
	counts = df[target].value_counts(dropna=False)
	total = counts.sum()
	return {
		"counts": counts.to_dict(),
		"proportion": (counts / total).round(4).to_dict(),
	}


def _duplicate_rows(df: pd.DataFrame) -> dict:
	dup_count = df.duplicated().sum()
	return {"count": int(dup_count), "status": "pass" if dup_count == 0 else "warn"}


def validate(config_path: str) -> None:
	config = load_config(config_path)
	ensure_dir(config["paths"]["output_dir"])
	logger = setup_logger(config["logging"]["log_file"])

	df = load_raw_dataset(config, logger)

	expected_columns = config["dataset"]["column_names"]
	target = config["dataset"]["target_column"]

	report = {
		"shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
		"schema": _check_schema(df, expected_columns),
		"missing_values": _missing_values(df),
		"duplicate_rows": _duplicate_rows(df),
		"label_distribution": _label_distribution(df, target),
	}

	save_json(report, config["validation"]["report_path"])

	logger.info("Validation complete. Report saved to %s", config["validation"]["report_path"])
	logger.info("Rows: %s, Columns: %s", df.shape[0], df.shape[1])
	logger.info("Duplicate rows: %s", report["duplicate_rows"]["count"])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Validate raw dataset")
	parser.add_argument("--config", default="config.yaml", help="Path to config file")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	validate(args.config)
