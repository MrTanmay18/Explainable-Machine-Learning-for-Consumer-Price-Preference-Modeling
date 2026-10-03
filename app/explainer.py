"""SHAP explanations for the saved XGBoost model.

* Local explanations are computed for the CURRENT request - never a fixed sample.
* XGBoost multiclass SHAP values are in raw log-odds (margin) space. For class k:
      margin_k = base_value_k + sum_j shap_{j,k}      and      p = softmax(margin)
  ``explain_local`` verifies this identity against ``predict_proba`` on every call.
* SHAP describes how the MODEL uses its inputs. It does not establish causality.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from app.predictor import RAW_FEATURES, ModelService, engineer_features
from app.schemas import MAX_AGE, MIN_AGE

logger = logging.getLogger("price_intel.explainer")

UNITS = "log-odds (raw XGBoost margin); class probabilities = softmax over the class margins"
DISCLAIMER = (
    "SHAP values describe how this model's output depends on the inputs. They show association "
    "learned from survey data, not proof that changing a factor would cause a different price preference."
)
MAX_REFERENCE_ROWS = 2000

# --------------------------------------------------------------------------- #
# Readable names
# --------------------------------------------------------------------------- #
SOURCE_LABELS = {
    "age_group": "Age Group",
    "gender": "Gender",
    "zone": "Zone",
    "occupation": "Occupation",
    "income_levels": "Income Level",
    "consume_frequency(weekly)": "Weekly Consumption",
    "current_brand": "Current Brand",
    "preferable_consumption_size": "Preferred Size",
    "awareness_of_other_brands": "Awareness of Other Brands",
    "reasons_for_choosing_brands": "Reason for Choosing Brand",
    "flavor_preference": "Flavor Preference",
    "purchase_channel": "Purchase Channel",
    "packaging_preference": "Packaging Preference",
    "health_concerns": "Health Concerns",
    "typical_consumption_situations": "Consumption Situation",
}
NUMERIC_LABELS = {
    "frequency_score": "Consumption Frequency Score",
    "awareness_score": "Brand Awareness Score",
    "cf_ab_score": "Frequency-to-Awareness Ratio (CFAB)",
    "zone_score": "Zone Score",
    "income_score": "Income Score",
    "zas_score": "Zone-Affluence Score (ZAS)",
    "bsi": "Brand Switching Indicator (BSI)",
}


@dataclass(frozen=True)
class FeatureMeta:
    transformed_name: str
    kind: str  # 'label' | 'onehot' | 'numeric'
    source: str  # column in the engineered frame
    category: Optional[str]  # only for one-hot dummies
    readable: str  # e.g. 'Occupation: Working Professional'


def build_feature_meta(model_service: ModelService) -> List[FeatureMeta]:
    """Derive readable names from the fitted preprocessor (nothing hard-coded per feature)."""
    pre = model_service.preprocessor
    onehot = pre.named_transformers_["onehot"]
    lookup: Dict[str, Tuple[str, str]] = {}
    drops = onehot.drop_idx_ if onehot.drop_idx_ is not None else [None] * len(onehot.categories_)
    for col, cats, drop in zip(onehot.feature_names_in_, onehot.categories_, drops):
        for i, cat in enumerate(cats):
            if drop is not None and i == drop:
                continue  # baseline category has no column
            lookup[f"onehot__{col}_{cat}"] = (col, str(cat))

    metas: List[FeatureMeta] = []
    for name in model_service.feature_names_out:
        if name.startswith("label__"):
            src = name[len("label__"):]
            metas.append(FeatureMeta(name, "label", src, None, SOURCE_LABELS.get(src, src)))
        elif name.startswith("onehot__"):
            col, cat = lookup[name]
            metas.append(FeatureMeta(name, "onehot", col, cat, f"{SOURCE_LABELS.get(col, col)}: {cat}"))
        elif name.startswith("numeric__"):
            src = name[len("numeric__"):]
            metas.append(FeatureMeta(name, "numeric", src, None, NUMERIC_LABELS.get(src, src)))
        else:  # pragma: no cover
            metas.append(FeatureMeta(name, "numeric", name, None, name))
    return metas


def _fmt_number(x: Any) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.2f}"


def display_for(meta: FeatureMeta, row: pd.Series) -> Tuple[str, str]:
    """(display_value, plot_label) for this respondent."""
    if meta.kind == "label":
        v = str(row[meta.source])
        return v, f"{meta.readable}: {v}"
    if meta.kind == "onehot":
        on = str(row[meta.source]) == meta.category
        if on:
            return "Yes", meta.readable
        return "No", f"{SOURCE_LABELS.get(meta.source, meta.source)}: not {meta.category}"
    v = _fmt_number(row[meta.source])
    return v, f"{meta.readable}: {v}"


# --------------------------------------------------------------------------- #
# SHAP helpers
# --------------------------------------------------------------------------- #
def normalize_shap(values: Any, n_features: int, n_classes: int) -> np.ndarray:
    """Return SHAP values as an array of shape (n_samples, n_features, n_classes).

    Handles the layouts different SHAP versions produce for multiclass models:
      * list of (n, F) arrays, one per class            (older SHAP)
      * ndarray (n, F, C)                               (SHAP >= 0.45, what the notebook used)
      * ndarray (C, n, F)
    """
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[1] == n_features and arr.shape[2] == n_classes:
        return arr
    if arr.shape[2] == n_features and arr.shape[0] == n_classes:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Unexpected SHAP output shape {arr.shape} for F={n_features}, C={n_classes}")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ShapService:
    def __init__(self, model_service: ModelService) -> None:
        self.ms = model_service
        self.meta = build_feature_meta(model_service)
        self.n_features = len(self.meta)
        self.n_classes = len(model_service.class_labels)

        self.explainer = shap.TreeExplainer(model_service.model)
        self.base_values, self.baseline_source = self._resolve_base_values()
        logger.info(
            "SHAP %s TreeExplainer ready (baseline %s from: %s)",
            shap.__version__, np.round(self.base_values, 3), self.baseline_source,
        )

        self._ref_lock = threading.Lock()
        self._ref: Optional[Dict[str, Any]] = None
        self.reference_info: Dict[str, Any] = {"available": False, "source": "none", "rows": 0, "is_demo": False}

    # ------------------------------------------------------------------ #
    def _resolve_base_values(self) -> Tuple[np.ndarray, str]:
        """Return per-class baselines that satisfy  margin = baseline + sum(SHAP).

        With some SHAP / XGBoost version combinations (multiclass + XGBoost>=2 per-class intercepts)
        ``TreeExplainer.expected_value`` is NOT the baseline XGBoost really uses, which silently breaks
        additivity. XGBoost's own ``pred_contribs`` bias column is exact, and the SHAP values themselves are
        identical, so we compare the two once at start-up and use the native bias if they disagree.
        """
        declared = np.atleast_1d(np.asarray(self.explainer.expected_value, dtype=float))
        if declared.shape[0] != self.n_classes:
            raise RuntimeError("SHAP expected_value does not match the number of classes.")
        booster = self.ms.model.get_booster()
        probe = np.zeros((1, self.n_features), dtype=float)  # the bias column does not depend on the input
        dm = xgb.DMatrix(probe, feature_names=booster.feature_names)
        with self.ms.lock:
            native = np.asarray(booster.predict(dm, pred_contribs=True))[0, :, -1].astype(float)  # (C,)
        if np.max(np.abs(declared - native)) <= 1e-3:
            return declared, "SHAP TreeExplainer expected_value (verified against XGBoost)"
        logger.warning(
            "TreeExplainer.expected_value %s != XGBoost native bias %s (shap %s, xgboost %s). "
            "Using the native bias so SHAP stays additive.",
            np.round(declared, 4), np.round(native, 4), shap.__version__, xgb.__version__,
        )
        return native, "XGBoost native bias (TreeExplainer.expected_value disagreed with the model in this SHAP/XGBoost version)"

    def _shap(self, Xt: pd.DataFrame) -> np.ndarray:
        with self.ms.lock:
            raw = self.explainer.shap_values(Xt)
        return normalize_shap(raw, self.n_features, self.n_classes)

    def class_index(self, label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        if label in self.ms.class_labels:
            return self.ms.class_labels.index(label)
        raise ValueError(f"Unknown price range '{label}'. Valid: {self.ms.class_labels}")

    # ------------------------------------------------------------------ #
    # LOCAL
    # ------------------------------------------------------------------ #
    def explain_local(self, frame: pd.DataFrame, target_class: Optional[str] = None, top_n: int = 5) -> Dict[str, Any]:
        Xt = self.ms.transform(frame)
        sv = self._shap(Xt)[0]  # (F, C)
        proba = self.ms.predict_proba(frame)[0]
        pred_idx = int(np.argmax(proba))
        tgt = self.class_index(target_class)
        tgt = pred_idx if tgt is None else tgt

        margins = self.base_values + sv.sum(axis=0)  # (C,)
        add_err = float(np.max(np.abs(softmax(margins) - proba)))
        if add_err > 1e-4:
            logger.warning("SHAP additivity error %.2e (expected ~0)", add_err)

        row = frame.iloc[0]
        contribs: List[Dict[str, Any]] = []
        for j, m in enumerate(self.meta):
            val = float(sv[j, tgt])
            disp, label = display_for(m, row)
            contribs.append(
                {
                    "feature": m.readable,
                    "transformed_name": m.transformed_name,
                    "display_value": disp,
                    "label": label,
                    "feature_value": float(Xt.iloc[0, j]),
                    "shap_value": val,
                    "direction": "increases" if val > 1e-9 else ("decreases" if val < -1e-9 else "neutral"),
                }
            )
        contribs.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        pos = sorted([c for c in contribs if c["shap_value"] > 0], key=lambda c: -c["shap_value"])
        neg = sorted([c for c in contribs if c["shap_value"] < 0], key=lambda c: c["shap_value"])

        total_pos = float(sum(c["shap_value"] for c in pos))
        total_neg = float(sum(c["shap_value"] for c in neg))
        order = self.ms.classes_in_price_order
        result = {
            "predicted_price_range": self.ms.class_labels[pred_idx],
            "explained_class_index": tgt,
            "explained_price_range": self.ms.class_labels[tgt],
            "units": UNITS,
            "baseline": float(self.base_values[tgt]),
            "baseline_source": self.baseline_source,
            "final_output": float(margins[tgt]),
            "explained_class_probability": float(proba[tgt]),
            "total_positive": total_pos,
            "total_negative": total_neg,
            "contributions": contribs,
            "top_positive": pos[:top_n],
            "top_negative": neg[:top_n],
            "class_summaries": [
                {
                    "class_index": i,
                    "price_range": self.ms.class_labels[i],
                    "baseline": float(self.base_values[i]),
                    "final_output": float(margins[i]),
                    "probability": float(proba[i]),
                }
                for i in order
            ],
            "additivity_max_error": add_err,
            "disclaimer": DISCLAIMER,
        }
        result["narrative"] = self._narrative(result, proba, pred_idx)
        return result

    def _narrative(self, r: Dict[str, Any], proba: np.ndarray, pred_idx: int) -> List[str]:
        tgt_label = r["explained_price_range"]
        lines: List[str] = []
        pred_label = self.ms.class_labels[pred_idx]
        lines.append(f"The model predicts the price range **{pred_label}** with {proba[pred_idx]:.1%} confidence.")
        if r["explained_class_index"] != pred_idx:
            lines.append(
                f"This explanation is for the alternative range **{tgt_label}** "
                f"(model probability {r['explained_class_probability']:.1%}), not the predicted one."
            )
        lines.append(
            f"For **{tgt_label}**, the model's raw score starts at a baseline of {r['baseline']:+.2f} "
            f"(its average output over the data the model was trained on) and ends at {r['final_output']:+.2f} "
            f"after considering this respondent's inputs. Factors pushing the score up added "
            f"{r['total_positive']:+.2f} in total, and factors pulling it down contributed {r['total_negative']:+.2f}."
        )
        for c in r["top_positive"][:3]:
            lines.append(f"**{c['label']}** raised the score for {tgt_label} by {c['shap_value']:+.2f}.")
        for c in r["top_negative"][:3]:
            lines.append(f"**{c['label']}** lowered the score for {tgt_label} by {abs(c['shap_value']):.2f}.")
        others = sorted(
            (s for s in r["class_summaries"] if s["class_index"] != pred_idx),
            key=lambda s: -s["probability"],
        )
        if others and r["explained_class_index"] == pred_idx:
            lines.append(
                f"The next most likely range is **{others[0]['price_range']}** ({others[0]['probability']:.1%})."
            )
        lines.append(DISCLAIMER)
        return lines

    # ------------------------------------------------------------------ #
    # GLOBAL (needs a reference sample of RAW survey rows)
    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Compute global SHAP once at start-up. Failure is non-fatal for /predict and /explain."""
        try:
            self._ensure_reference()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Global SHAP reference unavailable: %s", exc)

    def _ensure_reference(self) -> Dict[str, Any]:
        with self._ref_lock:
            if self._ref is not None:
                return self._ref
            path = self.ms.artifacts_dir / "reference_data.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python scripts/prepare_reference_data.py` to create it."
                )
            meta_file = self.ms.artifacts_dir / "reference_meta.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

            raw = pd.read_csv(path)
            missing = [c for c in RAW_FEATURES if c not in raw.columns]
            if missing:
                raise ValueError(f"reference_data.csv is missing columns: {missing}")
            raw = raw[RAW_FEATURES]
            raw = raw[raw["age"].between(MIN_AGE, MAX_AGE)]
            if len(raw) > MAX_REFERENCE_ROWS:
                raw = raw.sample(MAX_REFERENCE_ROWS, random_state=42)
            frame = engineer_features(raw.reset_index(drop=True))[self.ms.input_columns]
            Xt = self.ms.transform(frame)
            sv = self._shap(Xt)
            self._ref = {"Xt": Xt, "sv": sv}
            self.reference_info = {
                "available": True,
                "source": meta.get("source", "reference_data.csv"),
                "rows": int(len(Xt)),
                "is_demo": bool(meta.get("is_demo", False)),
            }
            logger.info("Global SHAP computed on %d reference rows (%s)", len(Xt), self.reference_info["source"])
            return self._ref

    def explain_global(self, target_class: Optional[str] = None, top_k: int = 15, max_points: int = 800) -> Dict[str, Any]:
        ref = self._ensure_reference()
        sv, Xt = ref["sv"], ref["Xt"]
        labels = self.ms.class_labels

        # Same definition as the notebook: mean |SHAP| over samples and classes.
        overall = np.abs(sv).mean(axis=(0, 2))  # (F,)
        per_class = np.abs(sv).mean(axis=0)  # (F, C)
        order = np.argsort(-overall)[:top_k]
        features = [
            {
                "feature": self.meta[j].readable,
                "transformed_name": self.meta[j].transformed_name,
                "mean_abs_shap": float(overall[j]),
                "per_class": {labels[c]: float(per_class[j, c]) for c in self.ms.classes_in_price_order},
            }
            for j in order
        ]

        c = self.class_index(target_class)
        if c is None:
            c = self.ms.classes_in_price_order[0]
        top_c = np.argsort(-per_class[:, c])[:top_k]
        n = min(max_points, len(Xt))
        beeswarm = {
            "price_range": labels[c],
            "feature_names": [self.meta[j].readable for j in top_c],
            "shap_values": np.round(sv[:n][:, top_c, c], 4).tolist(),
            "feature_values": np.round(Xt.iloc[:n, top_c].to_numpy(), 4).tolist(),
        }
        return {
            "reference_source": self.reference_info["source"],
            "reference_rows": self.reference_info["rows"],
            "is_demo_reference": self.reference_info["is_demo"],
            "units": UNITS,
            "features": features,
            "beeswarm": beeswarm,
        }
