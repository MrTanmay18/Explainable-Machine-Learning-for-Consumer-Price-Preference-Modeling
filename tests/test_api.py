"""Run with:  pytest -q   (from the project root)"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import engineer_features

VALID = {
    "age": 28, "gender": "M", "zone": "Urban", "occupation": "Working Professional",
    "income_levels": "<10L", "consume_frequency(weekly)": "0-2 times", "current_brand": "Newcomer",
    "preferable_consumption_size": "Small (250 ml)", "awareness_of_other_brands": "0 to 1",
    "reasons_for_choosing_brands": "Price", "flavor_preference": "Traditional",
    "purchase_channel": "Online", "packaging_preference": "Simple",
    "health_concerns": "Low (Not very concerned)", "typical_consumption_situations": "Active (eg. Sports, gym)",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan: model is loaded once
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] and r["explainer_ready"]


def test_feature_engineering_matches_notebook_logic():
    raw = pd.DataFrame([{**VALID, "zone": "Metro", "income_levels": "> 35L", "consume_frequency(weekly)": "5-7 times",
                         "awareness_of_other_brands": "2 to 4"}])
    f = engineer_features(raw).iloc[0]
    assert f["age_group"] == "26-35"
    assert f["frequency_score"] == 3 and f["awareness_score"] == 2
    assert f["cf_ab_score"] == 0.6            # round(3 / (2 + 3), 2)
    assert f["zone_score"] == 4 and f["income_score"] == 5 and f["zas_score"] == 20
    assert f["bsi"] == 1                      # Newcomer + Price
    assert "age" not in engineer_features(raw).columns


def test_predict_probabilities(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    j = r.json()
    assert abs(sum(p["probability"] for p in j["probabilities"]) - 1) < 1e-5
    assert [p["price_range"] for p in j["probabilities"]] == ["50-100", "100-150", "150-200", "200-250"]
    assert j["predicted_price_range"] == max(j["probabilities"], key=lambda p: p["probability"])["price_range"]
    assert j["engineered_features"]["zas_score"] == 3


def test_explain_is_additive_and_dynamic(client):
    a = client.post("/explain", json=VALID).json()
    b = client.post("/explain", json={**VALID, "age": 60, "zone": "Metro", "income_levels": "> 35L",
                                      "occupation": "Retired"}).json()
    for j in (a, b):
        assert j["additivity_max_error"] < 1e-4
        assert abs(j["baseline"] + sum(c["shap_value"] for c in j["contributions"]) - j["final_output"]) < 1e-4
        assert all(":" in c["label"] or " " in c["label"] for c in j["contributions"])
        assert not any("onehot__" in c["label"] or "label__" in c["label"] for c in j["contributions"])
    assert a["contributions"] != b["contributions"]  # explanation follows the input


def test_explain_target_class(client):
    j = client.post("/explain", params={"target_class": "50-100"}, json=VALID).json()
    assert j["explained_price_range"] == "50-100"
    assert client.post("/explain", params={"target_class": "bogus"}, json=VALID).status_code == 422


@pytest.mark.parametrize("patch", [{"age": 17}, {"age": 71}, {"zone": "Mars"}, {"zas_score": 12}, {"gender": "X"}])
def test_validation_errors(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_global_and_model_info(client):
    g = client.get("/global-explanation", params={"target_class": "200-250"}).json()
    assert g["features"][0]["mean_abs_shap"] >= g["features"][-1]["mean_abs_shap"]
    assert g["beeswarm"]["price_range"] == "200-250"
    info = client.get("/model-info").json()
    assert info["n_transformed_features"] == 31 and info["objective"] == "multi:softprob"
