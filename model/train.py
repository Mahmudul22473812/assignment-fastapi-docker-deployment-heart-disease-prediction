import argparse
import csv
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.schemas import FEATURE_NAMES, HeartFeatures

MODEL_PATH = Path(__file__).resolve().with_name("heart_model.joblib")
SEED = 42


def train(csv_path: Path) -> None:
    examples: dict[tuple[int | float, ...], int] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or ()) != {*FEATURE_NAMES, "target"}:
            raise ValueError("Expected the Kaggle heart.csv header with 13 features and target.")

        for row in reader:
            target = row.pop("target")
            if target not in {"0", "1"}:
                raise ValueError(f"Invalid target on CSV line {reader.line_num}: {target!r}")

            # In this Kaggle release, 0 matches UCI Cleveland disease presence.
            disease_present = int(target == "0")
            features = HeartFeatures.model_validate(row, strict=False).model_dump()
            values = tuple(features[name] for name in FEATURE_NAMES)
            if examples.setdefault(values, disease_present) != disease_present:
                raise ValueError(f"Conflicting labels on CSV line {reader.line_num}.")

    labels = list(examples.values())
    x_train, x_test, y_train, y_test = train_test_split(
        list(examples), labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    classifier = RandomForestClassifier(max_depth=6, random_state=SEED)
    classifier.fit(x_train, y_train)
    joblib.dump({"classifier": classifier, "features": FEATURE_NAMES}, MODEL_PATH, compress=3)

    print(f"Unique records: {len(examples)} ({len(x_train)} train, {len(x_test)} test)")
    print(f"Holdout accuracy: {classifier.score(x_test, y_test):.1%}")
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from the Kaggle heart.csv file.")
    parser.add_argument("csv_path", type=Path)
    train(parser.parse_args().csv_path)
