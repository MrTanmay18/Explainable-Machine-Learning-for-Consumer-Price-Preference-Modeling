"""Pydantic request / response schemas.

The category lists below mirror the categories learned by the encoders inside
the saved pipeline. ``ModelService`` cross-checks them at start-up
(``check_schema_against_model``) so a drift between this file and the model
artifact is caught immediately instead of surfacing as a confusing error.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union, get_args

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Allowed categories (raw survey values, exactly as used in training)
# --------------------------------------------------------------------------- #
Gender = Literal["F", "M"]
Zone = Literal["Metro", "Urban", "Semi-Urban", "Rural"]
Occupation = Literal["Student", "Working Professional", "Entrepreneur", "Retired"]
IncomeLevel = Literal["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L", "Not Reported"]
Frequency = Literal["0-2 times", "3-4 times", "5-7 times"]
Brand = Literal["Established", "Newcomer"]
Size = Literal["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"]
Awareness = Literal["0 to 1", "2 to 4", "above 4"]
Reason = Literal["Price", "Quality", "Brand Reputation", "Availability"]
Flavor = Literal["Traditional", "Exotic"]
Channel = Literal["Online", "Retail Store"]
Packaging = Literal["Simple", "Premium", "Eco-Friendly"]
Health = Literal["Low (Not very concerned)", "Medium (Moderately health-conscious)", "High (Very health-conscious)"]
Situation = Literal["Active (eg. Sports, gym)", "Casual (eg. At home)", "Social (eg. Parties)"]

# raw column name -> allowed values (used for the start-up consistency check)
CATEGORY_CHOICES: Dict[str, List[str]] = {
    "gender": list(get_args(Gender)),
    "zone": list(get_args(Zone)),
    "occupation": list(get_args(Occupation)),
    "income_levels": list(get_args(IncomeLevel)),
    "consume_frequency(weekly)": list(get_args(Frequency)),
    "current_brand": list(get_args(Brand)),
    "preferable_consumption_size": list(get_args(Size)),
    "awareness_of_other_brands": list(get_args(Awareness)),
    "reasons_for_choosing_brands": list(get_args(Reason)),
    "flavor_preference": list(get_args(Flavor)),
    "purchase_channel": list(get_args(Channel)),
    "packaging_preference": list(get_args(Packaging)),
    "health_concerns": list(get_args(Health)),
    "typical_consumption_situations": list(get_args(Situation)),
}

MIN_AGE, MAX_AGE = 18, 70


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
class ConsumerInput(BaseModel):
    """The 15 RAW survey inputs. Engineered features are NOT accepted."""

    model_config = ConfigDict(
        extra="forbid",  # engineered features must never be sent by the client
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 28,
                "gender": "M",
                "zone": "Urban",
                "occupation": "Working Professional",
                "income_levels": "<10L",
                "consume_frequency(weekly)": "0-2 times",
                "current_brand": "Newcomer",
                "preferable_consumption_size": "Small (250 ml)",
                "awareness_of_other_brands": "0 to 1",
                "reasons_for_choosing_brands": "Price",
                "flavor_preference": "Traditional",
                "purchase_channel": "Online",
                "packaging_preference": "Simple",
                "health_concerns": "Low (Not very concerned)",
                "typical_consumption_situations": "Active (eg. Sports, gym)",
            }
        },
    )

    age: int = Field(..., ge=MIN_AGE, le=MAX_AGE, description="Age in years (18-70).")
    gender: Gender
    zone: Zone
    occupation: Occupation
    income_levels: IncomeLevel
    consume_frequency: Frequency = Field(
        ...,
        validation_alias=AliasChoices("consume_frequency(weekly)", "consume_frequency"),
        serialization_alias="consume_frequency(weekly)",
    )
    current_brand: Brand
    preferable_consumption_size: Size
    awareness_of_other_brands: Awareness
    reasons_for_choosing_brands: Reason
    flavor_preference: Flavor
    purchase_channel: Channel
    packaging_preference: Packaging
    health_concerns: Health
    typical_consumption_situations: Situation

    def to_raw_dict(self) -> Dict[str, Union[int, str]]:
        """Return a dict keyed by the ORIGINAL training column names."""
        d = self.model_dump()
        d["consume_frequency(weekly)"] = d.pop("consume_frequency")
        return d


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class ClassProbability(BaseModel):
    class_index: int
    price_range: str
    probability: float


class ClassInfo(BaseModel):
    class_index: int
    price_range: str


class PredictionResponse(BaseModel):
    predicted_price_range: str
    predicted_class_index: int
    confidence: float = Field(..., description="Model probability of the predicted class (not a calibrated guarantee).")
    probabilities: List[ClassProbability] = Field(..., description="All classes, ordered from lowest to highest price range.")
    engineered_features: Dict[str, Union[int, float, str]] = Field(
        ..., description="Features generated automatically with the training-time logic."
    )
    warnings: List[str] = []


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Readable feature name, e.g. 'Occupation: Working Professional'.")
    transformed_name: str = Field(..., description="Name inside the model, e.g. 'onehot__occupation_Working Professional'.")
    display_value: str = Field(..., description="This respondent's value in human terms.")
    label: str = Field(..., description="Feature + value, ready for plot labels.")
    feature_value: float = Field(..., description="Numeric value actually fed to XGBoost after preprocessing.")
    shap_value: float
    direction: Literal["increases", "decreases", "neutral"]


class ClassSummary(BaseModel):
    class_index: int
    price_range: str
    baseline: float
    final_output: float
    probability: float


class ExplainResponse(BaseModel):
    predicted_price_range: str
    explained_class_index: int
    explained_price_range: str
    units: str
    baseline: float = Field(..., description="Expected raw model output for the explained class (SHAP base value).")
    baseline_source: str = Field(..., description="Where the baseline came from (validated against XGBoost at start-up).")
    final_output: float = Field(..., description="baseline + sum of all SHAP values (raw log-odds margin).")
    explained_class_probability: float
    total_positive: float
    total_negative: float
    contributions: List[FeatureContribution] = Field(..., description="All features, sorted by |SHAP| descending.")
    top_positive: List[FeatureContribution]
    top_negative: List[FeatureContribution]
    class_summaries: List[ClassSummary]
    narrative: List[str] = Field(..., description="Human-readable explanation generated from the SHAP values.")
    additivity_max_error: float = Field(
        ..., description="max |softmax(baseline + sum(SHAP)) - predict_proba()| across classes; ~0 means SHAP is consistent."
    )
    disclaimer: str


class GlobalFeature(BaseModel):
    feature: str
    transformed_name: str
    mean_abs_shap: float
    per_class: Dict[str, float]


class BeeswarmData(BaseModel):
    price_range: str
    feature_names: List[str]
    shap_values: List[List[float]] = Field(..., description="rows = reference respondents, cols = feature_names")
    feature_values: List[List[float]] = Field(..., description="Model-input values (for colouring), same shape.")


class GlobalExplanationResponse(BaseModel):
    reference_source: str
    reference_rows: int
    is_demo_reference: bool
    units: str
    features: List[GlobalFeature]
    beeswarm: BeeswarmData


class ModelInfoResponse(BaseModel):
    model_type: str
    objective: Optional[str]
    n_estimators: Optional[int]
    max_depth: Optional[int]
    learning_rate: Optional[float]
    classes: List[ClassInfo]
    n_transformed_features: int
    input_columns_in_training_order: List[str]
    transformed_feature_names: List[str]
    raw_input_features: List[str]
    engineered_features: List[str]
    categorical_options: Dict[str, List[str]]
    age_range: List[int]
    reported_training_metrics: Dict[str, Union[str, float, int]]
    reference_data: Dict[str, Union[str, int, bool]]
    library_versions: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_ready: bool
    reference_data_ready: bool
