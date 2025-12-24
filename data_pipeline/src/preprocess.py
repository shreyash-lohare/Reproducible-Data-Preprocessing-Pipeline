import argparse
import numpy as np
import pandas as pd

from src.utils import ensure_dir, ensure_parent_dir, load_config, load_raw_dataset, setup_logger


def handle_duplicates(df: pd.DataFrame, drop: bool, logger) -> pd.DataFrame:
	if not drop:
		return df
	before = df.shape[0]
	df = df.drop_duplicates().reset_index(drop=True)
	logger.info("Dropped %s duplicate rows", before - df.shape[0])
	return df


def handle_missing(df: pd.DataFrame, numeric_cols, categorical_cols, num_strategy, cat_strategy, logger):
	before = df.shape[0]
	if num_strategy == "drop" or cat_strategy == "drop":
		df = df.dropna(subset=numeric_cols + categorical_cols)
		logger.info("Dropped %s rows with missing values", before - df.shape[0])
		return df

	if num_strategy == "median":
		df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
	elif num_strategy == "mean":
		df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
	elif num_strategy == "most_frequent":
		df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
	else:
		raise ValueError(f"Unsupported numeric missing value strategy: {num_strategy}")

	if cat_strategy == "most_frequent":
		df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode().iloc[0]))
	elif cat_strategy == "constant":
		df[categorical_cols] = df[categorical_cols].fillna("missing")
	else:
		raise ValueError(f"Unsupported categorical missing value strategy: {cat_strategy}")

	return df


def encode_categoricals(df: pd.DataFrame, categorical_cols, encoding: str, logger) -> pd.DataFrame:
	if encoding == "onehot":
		df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
		logger.info("Applied one-hot encoding to %s categorical columns", len(categorical_cols))
		return df

	if encoding == "ordinal":
		for col in categorical_cols:
			df[col], _ = pd.factorize(df[col], sort=True)
		logger.info("Applied ordinal encoding to %s categorical columns", len(categorical_cols))
		return df

	raise ValueError(f"Unsupported categorical encoding: {encoding}")


def normalize_numeric(df: pd.DataFrame, numeric_cols, logger) -> pd.DataFrame:
	for col in numeric_cols:
		mean = df[col].mean()
		std = df[col].std()
		if std == 0 or pd.isna(std):
			df[col] = 0
		else:
			df[col] = (df[col] - mean) / std
	logger.info("Normalized %s numeric columns", len(numeric_cols))
	return df


def split_dataset(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
	rng = np.random.RandomState(seed)
	indices = rng.permutation(len(df))
	n_test = int(len(df) * test_size)
	n_val = int(len(df) * val_size)

	test_idx = indices[:n_test]
	val_idx = indices[n_test : n_test + n_val]
	train_idx = indices[n_test + n_val :]

	return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def preprocess(config_path: str) -> None:
	config = load_config(config_path)
	paths = config["paths"]
	ensure_dir(paths["output_dir"])
	logger = setup_logger(config["logging"]["log_file"])

	df = load_raw_dataset(config, logger)
	target = config["dataset"]["target_column"]
	cat_cols = config["dataset"]["categorical_columns"]
	num_cols = config["dataset"]["numeric_columns"]

	rows_before = df.shape[0]

	df = handle_duplicates(df, config["preprocessing"].get("drop_duplicates", True), logger)
	df = handle_missing(
		df,
		num_cols,
		cat_cols,
		config["preprocessing"]["missing_value_strategy"].get("numeric", "median"),
		config["preprocessing"]["missing_value_strategy"].get("categorical", "most_frequent"),
		logger,
	)

	encoding = config["preprocessing"]["categorical_encoding"]
	df = encode_categoricals(df, cat_cols, encoding, logger)

	if config["preprocessing"].get("normalize_numeric", True):
		df = normalize_numeric(df, num_cols, logger)

	df = df.dropna(subset=[target])

	val_size = config["preprocessing"].get("val_size", 0.1)
	test_size = config["preprocessing"].get("test_size", 0.2)
	seed = config["preprocessing"].get("random_seed", 42)

	train_df, val_df, test_df = split_dataset(df, val_size, test_size, seed)

	logger.info(
		"Split data into train/val/test shapes: %s / %s / %s",
		train_df.shape,
		val_df.shape,
		test_df.shape,
	)
	logger.info("Rows before cleaning: %s, after: %s", rows_before, df.shape[0])

	ensure_parent_dir(paths["processed_train"])
	train_df.to_csv(paths["processed_train"], index=False)
	val_df.to_csv(paths["processed_val"], index=False)
	test_df.to_csv(paths["processed_test"], index=False)

	logger.info("Saved processed train to %s", paths["processed_train"])
	logger.info("Saved processed val to %s", paths["processed_val"])
	logger.info("Saved processed test to %s", paths["processed_test"])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess dataset")
	parser.add_argument("--config", default="config.yaml", help="Path to config file")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	preprocess(args.config)
