"""Create artifacts/reference_data.csv - the RAW-feature sample used for GLOBAL SHAP plots.

Real data (recommended):
    python scripts/prepare_reference_data.py --csv survey_results.csv

It repeats the notebook's cleaning (sections 3 and 5) and the same
train_test_split(test_size=0.25, random_state=42), then samples from the TEST
split, as your notebook's SHAP section did. The model is NOT retrained.

Demo data (when you do not have the CSV at hand):
    python scripts/prepare_reference_data.py
Synthetic respondents are generated and the metadata flags them as a demo, so the UI warns that
global plots do not reflect your real survey.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.predictor import RAW_FEATURES  # noqa: E402
from app.schemas import CATEGORY_CHOICES  # noqa: E402

OUT_DIR = ROOT / "artifacts"


# --------------------------------------------------------------------------- #
def clean_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """Sections 3.1-3.4 and 5.1 of the notebook (same order, same rules)."""
    df = df.drop_duplicates()
    df = df[df["age"] < 100]
    df["income_levels"] = df["income_levels"].fillna("Not Reported")
    cf_col, pc_col = "consume_frequency(weekly)", "purchase_channel"
    df[cf_col] = df[cf_col].fillna(df[cf_col].mode()[0])
    df[pc_col] = df[pc_col].fillna(df[pc_col].mode()[0])
    df["zone"] = df["zone"].replace({"Metor": "Metro", "urbna": "Urban"})
    df["current_brand"] = df["current_brand"].replace({"newcomer": "Newcomer", "Establishd": "Established"})
    # students aged 56-70 were removed in section 5.1
    df = df[~((df["age"].between(56, 70)) & (df["occupation"] == "Student"))]
    # the saved encoder has no '70+' group, so training data had no age > 70
    df = df[df["age"] <= 70]
    return df


def from_survey(csv_path: Path, n: int) -> tuple[pd.DataFrame, dict]:
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df = clean_like_notebook(df)
    _, test = train_test_split(df, test_size=0.25, random_state=42)  # identical row selection to the notebook
    sample = test.sample(min(n, len(test)), random_state=42)[RAW_FEATURES]
    meta = {
        "source": f"{csv_path.name} (notebook test split, random_state=42)",
        "is_demo": False,
        "rows": int(len(sample)),
    }
    return sample, meta


def synthetic(n: int, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Plausible but ARTIFICIAL respondents (zone/brand marginals taken from the notebook output)."""
    rng = np.random.default_rng(seed)
    C = CATEGORY_CHOICES
    age = np.clip(np.round(18 + rng.gamma(2.0, 7.5, n)), 18, 70).astype(int)

    occ = []
    for a in age:
        if a <= 25:
            p = [0.45, 0.45, 0.10, 0.0]  # Student, WP, Entrepreneur, Retired
        elif a <= 35:
            p = [0.10, 0.65, 0.25, 0.0]
        elif a <= 55:
            p = [0.0, 0.65, 0.30, 0.05]
        else:
            p = [0.0, 0.30, 0.20, 0.50]
        occ.append(rng.choice(["Student", "Working Professional", "Entrepreneur", "Retired"], p=p))
    occ = np.array(occ)

    income = rng.choice(C["income_levels"][:5], n)
    income = np.where(rng.random(n) < 0.10, "Not Reported", income)
    income = np.where(occ == "Student", "Not Reported", income)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": rng.choice(C["gender"], n),
            "zone": rng.choice(["Metro", "Urban", "Semi-Urban", "Rural"], n, p=[0.397, 0.357, 0.176, 0.070]),
            "occupation": occ,
            "income_levels": income,
            "consume_frequency(weekly)": rng.choice(C["consume_frequency(weekly)"], n, p=[0.35, 0.40, 0.25]),
            "current_brand": rng.choice(["Established", "Newcomer"], n, p=[0.516, 0.484]),
            "preferable_consumption_size": rng.choice(C["preferable_consumption_size"], n),
            "awareness_of_other_brands": rng.choice(C["awareness_of_other_brands"], n, p=[0.30, 0.45, 0.25]),
            "reasons_for_choosing_brands": rng.choice(C["reasons_for_choosing_brands"], n),
            "flavor_preference": rng.choice(C["flavor_preference"], n),
            "purchase_channel": rng.choice(C["purchase_channel"], n),
            "packaging_preference": rng.choice(C["packaging_preference"], n),
            "health_concerns": rng.choice(C["health_concerns"], n),
            "typical_consumption_situations": rng.choice(C["typical_consumption_situations"], n),
        }
    )[RAW_FEATURES]
    meta = {
        "source": "SYNTHETIC demo respondents (not your survey data)",
        "is_demo": True,
        "rows": int(len(df)),
    }
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="Path to your survey_results.csv")
    ap.add_argument("--rows", type=int, default=2000, help="Reference sample size (default 2000)")
    args = ap.parse_args()

    df, meta = from_survey(args.csv, args.rows) if args.csv else synthetic(args.rows)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "reference_data.csv", index=False)
    (OUT_DIR / "reference_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {OUT_DIR / 'reference_data.csv'}")
    print(f"Source: {meta['source']}")
    if meta["is_demo"]:
        print("NOTE: demo data. Re-run with --csv survey_results.csv for real global SHAP plots.")


if __name__ == "__main__":
    main()
