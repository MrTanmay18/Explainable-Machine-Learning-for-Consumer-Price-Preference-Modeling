"""Model loading, feature engineering and prediction.

IMPORTANT: ``engineer_features`` reproduces, line for line, the feature
engineering from the training notebook (sections 4.x). Do not "improve" it:
the saved pipeline was fitted on features produced by exactly this logic.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from app.schemas import CATEGORY_CHOICES, MAX_AGE, MIN_AGE

logger = logging.getLogger("price_intel.predictor")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "xgboost_final_model.joblib"

# --------------------------------------------------------------------------- #
# Feature-engineering constants  (copied from the notebook)
# --------------------------------------------------------------------------- #
AGE_BINS = [17, 25, 35, 45, 55, 70, float("inf")]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-70", "70+"]

FREQUENCY_MAP = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
AWARENESS_MAP = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
ZONE_MAP = {"Urban": 3, "Metro": 4, "Rural": 1, "Semi-Urban": 2}
INCOME_LEVEL_MAP = {"<10L": 1, "10L - 15L": 2, "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5, "Not Reported": 0}

RAW_FEATURES: List[str] = [
    "age", "gender", "zone", "occupation", "income_levels",
    "consume_frequency(weekly)", "current_brand", "preferable_consumption_size",
    "awareness_of_other_brands", "reasons_for_choosing_brands", "flavor_preference",
    "purchase_channel", "packaging_preference", "health_concerns",
    "typical_consumption_situations",
]
ENGINEERED_FEATURES: List[str] = [
    "age_group", "frequency_score", "awareness_score", "cf_ab_score",
    "zone_score", "income_score", "zas_score", "bsi",
]

# Fallback if label_mapping.json is missing: sklearn LabelEncoder = alphabetical order.
DEFAULT_CLASS_LABELS = ["100-150", "150-200", "200-250", "50-100"]

# Friendlier (semantic) ordering of dropdown options for the UI only.
UI_ORDER = {
    "income_levels": ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L", "Not Reported"],
    "consume_frequency(weekly)": ["0-2 times", "3-4 times", "5-7 times"],
    "preferable_consumption_size": ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"],
    "awareness_of_other_brands": ["0 to 1", "2 to 4", "above 4"],
    "health_concerns": ["Low (Not very concerned)", "Medium (Moderately health-conscious)", "High (Very health-conscious)"],
    "zone": ["Metro", "Urban", "Semi-Urban", "Rural"],
}


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the training-time feature engineering to RAW survey rows.

    Input must contain ``RAW_FEATURES``. Output has ``age`` replaced by
    ``age_group`` plus the 7 engineered numeric features.
    """
    df = raw.copy()

    # 4.1 Age binning (the raw 'age' column is dropped afterwards, as in training)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    if df["age_group"].isna().any() or (df["age_group"].astype(str) == "70+").any():
        raise ValueError(
            f"Age must be between {MIN_AGE} and {MAX_AGE}: the model's encoder was trained without a '70+' group."
        )
    df["age_group"] = df["age_group"].astype(str)

    # 4.2 Ordinal scores
    df["frequency_score"] = df["consume_frequency(weekly)"].map(FREQUENCY_MAP)
    df["awareness_score"] = df["awareness_of_other_brands"].map(AWARENESS_MAP)

    # 4.3 Frequency-to-awareness ratio score
    df["cf_ab_score"] = (df["frequency_score"] / (df["awareness_score"] + df["frequency_score"])).round(2)

    # 4.4 Zone & income scores
    df["zone_score"] = df["zone"].map(ZONE_MAP)
    df["income_score"] = df["income_levels"].map(INCOME_LEVEL_MAP)

    # 4.5 Zone-Affluence Score
    df["zas_score"] = df["zone_score"] * df["income_score"]

    # 4.6 Brand Switching Indicator
    df["bsi"] = (
        (df["current_brand"] != "Established")
        & (df["reasons_for_choosing_brands"].isin(["Price", "Quality"]))
    ).astype(int)

    unmapped = [c for c in ("frequency_score", "awareness_score", "zone_score", "income_score") if df[c].isna().any()]
    if unmapped:
        raise ValueError(f"Unrecognised category value(s) produced missing scores: {unmapped}")

    return df.drop(columns=["age"])


def _price_sort_key(label: str) -> float:
    """'50-100' -> 50 so labels sort by price, not alphabetically."""
    try:
        return float(label.split("-")[0])
    except ValueError:  # pragma: no cover
        return float("inf")


class ModelService:
    """Loads the saved pipeline ONCE and exposes prediction helpers."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        model_path = self.artifacts_dir / "xgboost_final_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        logger.info("Loading pipeline from %s", model_path)
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        # EXACT training feature order, taken from the fitted preprocessor itself.
        self.input_columns: List[str] = list(self.preprocessor.feature_names_in_)
        self.feature_names_out: List[str] = list(self.preprocessor.get_feature_names_out())

        self.class_labels: List[str] = self._load_class_labels()
        self.lock = threading.Lock()  # serialise calls into the booster
        self.metadata = self._load_json("model_metadata.json")

        if len(self.model.classes_) != len(self.class_labels):
            raise RuntimeError("label_mapping.json does not match the number of model classes.")
        if self.model.n_features_in_ != len(self.feature_names_out):
            raise RuntimeError("Model / preprocessor feature-count mismatch.")
        self.check_schema_against_model()
        logger.info("Model ready: %d classes, %d features", len(self.class_labels), len(self.feature_names_out))

    # ------------------------------------------------------------------ #
    def _load_json(self, name: str) -> Dict[str, Any]:
        p = self.artifacts_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _load_class_labels(self) -> List[str]:
        mapping = self._load_json("label_mapping.json")
        mapping = {k: v for k, v in mapping.items() if not k.startswith("_")}
        if not mapping:
            logger.warning("label_mapping.json missing - using alphabetical default %s", DEFAULT_CLASS_LABELS)
            return list(DEFAULT_CLASS_LABELS)
        return [mapping[str(i)] for i in range(len(mapping))]

    @property
    def classes_in_price_order(self) -> List[int]:
        return sorted(range(len(self.class_labels)), key=lambda i: _price_sort_key(self.class_labels[i]))

    # ------------------------------------------------------------------ #
    def categorical_options(self) -> Dict[str, List[str]]:
        """Categories learned by the encoders (raw columns only)."""
        opts: Dict[str, List[str]] = {}
        for name in ("label", "onehot"):
            enc = self.preprocessor.named_transformers_[name]
            for col, cats in zip(enc.feature_names_in_, enc.categories_):
                opts[col] = [str(c) for c in cats]
        opts.pop("age_group", None)  # derived from age, not user-entered
        for col, order in UI_ORDER.items():
            if col in opts and set(order) == set(opts[col]):
                opts[col] = list(order)
        return opts

    def check_schema_against_model(self) -> None:
        learned = self.categorical_options()
        problems = []
        for col, allowed in CATEGORY_CHOICES.items():
            if set(allowed) != set(learned.get(col, [])):
                problems.append(f"{col}: schema={sorted(allowed)} model={sorted(learned.get(col, []))}")
        if problems:
            raise RuntimeError("schemas.py is out of sync with the model's encoders:\n" + "\n".join(problems))

    # ------------------------------------------------------------------ #
    def build_frame(self, raw: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """RAW inputs -> engineered frame in the exact training column order."""
        raw_df = pd.DataFrame([raw]) if isinstance(raw, dict) else raw
        raw_df = raw_df[RAW_FEATURES]
        engineered = engineer_features(raw_df)
        return engineered[self.input_columns]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        with self.lock:
            arr = self.preprocessor.transform(frame)
        return pd.DataFrame(np.asarray(arr, dtype=float), columns=self.feature_names_out, index=frame.index)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        with self.lock:
            return self.pipeline.predict_proba(frame)

    def predict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        frame = self.build_frame(raw)
        proba = self.predict_proba(frame)[0]
        idx = int(np.argmax(proba))
        probs = [
            {"class_index": i, "price_range": self.class_labels[i], "probability": float(proba[i])}
            for i in self.classes_in_price_order
        ]
        engineered = {
            c: (frame.iloc[0][c].item() if hasattr(frame.iloc[0][c], "item") else frame.iloc[0][c])
            for c in ENGINEERED_FEATURES
        }
        return {
            "predicted_price_range": self.class_labels[idx],
            "predicted_class_index": idx,
            "confidence": float(proba[idx]),
            "probabilities": probs,
            "engineered_features": engineered,
            "warnings": self.input_warnings(raw),
            "_frame": frame,
        }

    @staticmethod
    def input_warnings(raw: Dict[str, Any]) -> List[str]:
        warns = []
        if raw.get("occupation") == "Student" and raw.get("age", 0) >= 56:
            warns.append(
                "Students aged 56-70 were removed as illogical records during training, "
                "so the model has not seen this combination; treat the result with caution."
            )
        return warns

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        import sklearn
        import xgboost

        params = self.model.get_params()
        metrics = {k: v for k, v in self.metadata.items() if k not in {"model_name"}}
        return {
            "model_type": type(self.model).__name__,
            "objective": params.get("objective"),
            "n_estimators": params.get("n_estimators"),
            "max_depth": params.get("max_depth"),
            "learning_rate": params.get("learning_rate"),
            "classes": [
                {"class_index": i, "price_range": self.class_labels[i]} for i in self.classes_in_price_order
            ],
            "n_transformed_features": len(self.feature_names_out),
            "input_columns_in_training_order": self.input_columns,
            "transformed_feature_names": self.feature_names_out,
            "raw_input_features": RAW_FEATURES,
            "engineered_features": ENGINEERED_FEATURES,
            "categorical_options": self.categorical_options(),
            "age_range": [MIN_AGE, MAX_AGE],
            "reported_training_metrics": metrics,
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
