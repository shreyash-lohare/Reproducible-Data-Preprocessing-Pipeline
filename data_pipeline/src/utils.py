import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_config(config_path: str) -> Dict:
	"""Load YAML config and return as dict."""
	with open(config_path, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)
	if not isinstance(config, dict):
		raise ValueError("Config file is empty or invalid.")
	return config


def ensure_parent_dir(path: str) -> None:
	Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str) -> None:
	Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: str) -> logging.Logger:
	ensure_parent_dir(log_path)
	logger = logging.getLogger("data_pipeline")
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		handler = logging.FileHandler(log_path)
		formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger.addHandler(console)
	return logger


def load_raw_dataset(config: Dict, logger: logging.Logger | None = None) -> pd.DataFrame:
	dataset_cfg = config["dataset"]
	columns: List[str] = dataset_cfg["column_names"]
	train_path = dataset_cfg["raw_train_path"]
	test_path = dataset_cfg["raw_test_path"]

	df_train = pd.read_csv(
		train_path,
		names=columns,
		na_values=["?"],
		skipinitialspace=True,
		header=None,
	)
	df_test = pd.read_csv(
		test_path,
		names=columns,
		na_values=["?"],
		skipinitialspace=True,
		header=None,
		skiprows=1,
	)

	target = dataset_cfg["target_column"]
	if target in df_test.columns:
		df_test[target] = df_test[target].astype(str).str.replace(".", "", regex=False).str.strip()

	df = pd.concat([df_train, df_test], ignore_index=True)
	df = _strip_strings(df)
	if logger:
		logger.info("Loaded raw dataset: %s rows, %s columns", df.shape[0], df.shape[1])
	return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.select_dtypes(include=["object", "string"]).columns:
		df[col] = df[col].astype(str).str.strip()
		df[col] = df[col].replace({"?": pd.NA})
	return df


def save_json(data: Dict, path: str) -> None:
	ensure_parent_dir(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)
