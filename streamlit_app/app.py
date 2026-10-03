"""Consumer Price Preference Intelligence - Streamlit frontend.

Talks to the FastAPI backend over HTTP only (no model code lives here).
    streamlit run streamlit_app/app.py
Set API_URL to point at a different backend (default http://127.0.0.1:8000).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60

st.set_page_config(page_title="Consumer Price Preference Intelligence", layout="wide", page_icon=None)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1300px;}
      .app-title {font-size: 2.3rem; font-weight: 750; margin-bottom: 0; letter-spacing: -0.5px;}
      .app-sub {color: #6b7280; margin-top: 0.2rem; margin-bottom: 1.2rem;}
      .result-card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.1rem 1.4rem;
                    background: rgba(99,102,241,.07);}
      .result-label {font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: #6b7280;}
      .result-value {font-size: 2.2rem; font-weight: 750; line-height: 1.15;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    pass


def _raise_for(resp: requests.Response) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if isinstance(detail, list):  # pydantic validation errors
        detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    raise ApiError(f"API returned {resp.status_code}: {detail}")


def api_get(path: str, params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn app.main:app --port 8000") from exc
    _raise_for(r)
    return r.json()


def api_post(path: str, payload: dict, params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn app.main:app --port 8000") from exc
    _raise_for(r)
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def load_model_info(api_url: str) -> Dict[str, Any]:  # api_url keeps the cache key per backend
    return api_get("/model-info")


@st.cache_data(ttl=300, show_spinner=False)
def load_global(api_url: str, target_class: str) -> Dict[str, Any]:
    return api_get("/global-explanation", {"target_class": target_class, "top_k": 15, "max_points": 800})


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def waterfall_figure(exp: Dict[str, Any], max_display: int = 12):
    contribs = exp["contributions"]  # already sorted by |SHAP|
    e = shap.Explanation(
        values=np.array([c["shap_value"] for c in contribs]),
        base_values=exp["baseline"],
        data=None,
        feature_names=[c["label"] for c in contribs],
    )
    plt.close("all")
    shap.plots.waterfall(e, max_display=max_display, show=False)
    return plt.gcf()


def beeswarm_figure(b: Dict[str, Any]):
    vals = np.array(b["shap_values"])
    e = shap.Explanation(
        values=vals,
        base_values=np.zeros(len(vals)),
        data=np.array(b["feature_values"]),
        feature_names=b["feature_names"],
    )
    plt.close("all")
    shap.plots.beeswarm(e, max_display=len(b["feature_names"]), show=False)
    return plt.gcf()


def contrib_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Factor": [c["label"] for c in items],
            "Impact (log-odds)": [round(c["shap_value"], 3) for c in items],
        }
    )


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


# --------------------------------------------------------------------------- #
# Header + sidebar
# --------------------------------------------------------------------------- #
st.markdown('<div class="app-title">Consumer Price Preference Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Behavioral modeling of consumer price sensitivity using explainable machine learning '
    "(XGBoost + SHAP)</div>",
    unsafe_allow_html=True,
)

try:
    info = load_model_info(API_URL)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

opts = info["categorical_options"]
min_age, max_age = info["age_range"]
class_labels = [c["price_range"] for c in info["classes"]]  # price order

with st.sidebar:
    st.subheader("System")
    st.success(f"API connected: {API_URL}")
    m = info["reported_training_metrics"]
    st.caption(
        f"**Model:** {info['model_type']} ({info['n_estimators']} trees, depth {info['max_depth']})  \n"
        f"**Reported test accuracy:** {m.get('test_accuracy', 'n/a')}  \n"
        f"**Reported macro F1:** {m.get('macro_f1', 'n/a')}  \n"
        f"**Classes:** {', '.join(class_labels)}"
    )
    with st.expander("Library versions"):
        st.json(info["library_versions"])
    st.subheader("How to read this")
    st.caption(
        "SHAP values are in **log-odds** units. A positive value pushes the model toward the explained price range, "
        "a negative value pushes away. They explain the **model**, they do not prove cause and effect."
    )

# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
st.subheader("Consumer profile")
with st.form("consumer_form"):
    r1 = st.columns(4)
    age = r1[0].number_input("Age", min_value=min_age, max_value=max_age, value=28, step=1)
    gender = r1[1].selectbox("Gender", opts["gender"], index=default_index(opts["gender"], "M"))
    zone = r1[2].selectbox("Zone", opts["zone"], index=default_index(opts["zone"], "Urban"))
    occupation = r1[3].selectbox(
        "Occupation", opts["occupation"], index=default_index(opts["occupation"], "Working Professional")
    )

    r2 = st.columns(4)
    income = r2[0].selectbox("Income Level (In L)", opts["income_levels"], index=0)
    freq = r2[1].selectbox("Consume Frequency (weekly)", opts["consume_frequency(weekly)"], index=0)
    brand = r2[2].selectbox("Current Brand", opts["current_brand"], index=default_index(opts["current_brand"], "Newcomer"))
    size = r2[3].selectbox("Preferable Consumption Size", opts["preferable_consumption_size"], index=0)

    r3 = st.columns(4)
    aware = r3[0].selectbox("Awareness of other brands", opts["awareness_of_other_brands"], index=0)
    reason = r3[1].selectbox(
        "Reasons for choosing brands",
        opts["reasons_for_choosing_brands"],
        index=default_index(opts["reasons_for_choosing_brands"], "Price"),
    )
    flavor = r3[2].selectbox(
        "Flavor Preference", opts["flavor_preference"], index=default_index(opts["flavor_preference"], "Traditional")
    )
    channel = r3[3].selectbox(
        "Purchase Channel", opts["purchase_channel"], index=default_index(opts["purchase_channel"], "Online")
    )

    r4 = st.columns(4)
    packaging = r4[0].selectbox(
        "Packaging Preference", opts["packaging_preference"], index=default_index(opts["packaging_preference"], "Simple")
    )
    health = r4[1].selectbox("Health Concerns", opts["health_concerns"], index=0)
    situation = r4[2].selectbox(
        "Typical Consumption Situations",
        opts["typical_consumption_situations"],
        index=default_index(opts["typical_consumption_situations"], "Active (eg. Sports, gym)"),
    )
    submitted = st.form_submit_button("Predict Price Range", type="primary")

if submitted:
    payload = {
        "age": int(age),
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income,
        "consume_frequency(weekly)": freq,
        "current_brand": brand,
        "preferable_consumption_size": size,
        "awareness_of_other_brands": aware,
        "reasons_for_choosing_brands": reason,
        "flavor_preference": flavor,
        "purchase_channel": channel,
        "packaging_preference": packaging,
        "health_concerns": health,
        "typical_consumption_situations": situation,
    }
    try:
        with st.spinner("Predicting and computing SHAP explanation ..."):
            prediction = api_post("/predict", payload)
            explanation = api_post("/explain", payload)
        st.session_state.update(payload=payload, prediction=prediction, explanation=explanation)
    except ApiError as exc:
        st.session_state.pop("prediction", None)
        st.error(str(exc))

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if "prediction" not in st.session_state:
    st.info("Fill in the consumer profile and press **Predict Price Range**.")
    st.stop()

pred = st.session_state["prediction"]
payload = st.session_state["payload"]
exp = st.session_state["explanation"]

tab_pred, tab_why, tab_global, tab_bee = st.tabs(
    ["Prediction", "Why this prediction?", "Global importance", "Beeswarm"]
)

# ---- Prediction ------------------------------------------------------------
with tab_pred:
    for w in pred.get("warnings", []):
        st.warning(w)
    ranked = sorted(pred["probabilities"], key=lambda p: -p["probability"])
    c1, c2, c3 = st.columns([1.3, 1, 1])
    c1.markdown(
        f'<div class="result-card"><div class="result-label">Predicted price range</div>'
        f'<div class="result-value">{pred["predicted_price_range"]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.metric("Confidence", f"{pred['confidence']:.1%}", help="Model-estimated probability of the predicted class; not a calibrated guarantee.")
    c3.metric("Runner-up", ranked[1]["price_range"], f"{ranked[1]['probability']:.1%}", delta_color="off")

    st.markdown("##### Probability distribution across price ranges")
    pdf = pd.DataFrame(pred["probabilities"])
    pdf["predicted"] = pdf["price_range"] == pred["predicted_price_range"]
    chart = (
        alt.Chart(pdf)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("price_range:N", sort=class_labels, title="Price range"),
            y=alt.Y("probability:Q", axis=alt.Axis(format="%"), title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.predicted, alt.value("#6366f1"), alt.value("#c7c9d9")),
            tooltip=["price_range", alt.Tooltip("probability:Q", format=".2%")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("Features generated automatically by the API (not entered by you)"):
        ef = pred["engineered_features"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": ["Age group", "Frequency score", "Awareness score", "CFAB score", "Zone score",
                                "Income score", "Zone-Affluence Score (ZAS)", "Brand Switching Indicator (BSI)"],
                    "Value": [str(ef[k]) for k in ["age_group", "frequency_score", "awareness_score", "cf_ab_score",
                                                   "zone_score", "income_score", "zas_score", "bsi"]],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Computed server-side with the same logic used during training.")

# ---- Why this prediction? --------------------------------------------------
with tab_why:
    st.markdown("#### Why this prediction?")
    chosen = st.selectbox(
        "Explain which price range?",
        class_labels,
        index=class_labels.index(exp["explained_price_range"]),
        help="Defaults to the predicted range. Pick another to see what pushed the model toward or away from it.",
    )
    if chosen != exp["explained_price_range"]:
        try:
            exp = api_post("/explain", payload, params={"target_class": chosen})
            st.session_state["explanation"] = exp
        except ApiError as exc:
            st.error(str(exc))

    for line in exp["narrative"][:-1]:
        st.markdown(f"- {line}")

    st.markdown("##### SHAP waterfall (this respondent)")
    fig = waterfall_figure(exp)
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption(
        f"Starts at the baseline E[f(X)] = {exp['baseline']:+.2f} and adds each feature's contribution to reach "
        f"f(x) = {exp['final_output']:+.2f} (raw log-odds for {exp['explained_price_range']}; "
        f"model probability {exp['explained_class_probability']:.1%})."
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Baseline", f"{exp['baseline']:+.2f}")
    b2.metric("Net contribution", f"{exp['total_positive'] + exp['total_negative']:+.2f}",
              f"+{exp['total_positive']:.2f} / {exp['total_negative']:.2f}", delta_color="off")
    b3.metric("Final model output", f"{exp['final_output']:+.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown(f"##### Top factors pushing **toward** {exp['explained_price_range']}")
        st.dataframe(contrib_table(exp["top_positive"]), hide_index=True, width="stretch")
    with right:
        st.markdown(f"##### Top factors pushing **away from** {exp['explained_price_range']}")
        st.dataframe(contrib_table(exp["top_negative"]), hide_index=True, width="stretch")

    st.caption(exp["narrative"][-1])
    st.caption(
        f"SHAP consistency check, max |softmax(baseline + sum SHAP) - predict_proba|: {exp['additivity_max_error']:.1e}. "
        f"Baseline source: {exp['baseline_source']}."
    )

# ---- Global importance -----------------------------------------------------
with tab_global:
    st.markdown("#### Global SHAP feature importance")
    try:
        g = load_global(API_URL, exp["explained_price_range"])
        if g["is_demo_reference"]:
            st.warning(
                f"Reference sample: **{g['reference_source']}**. Global plots below show how the saved model behaves on "
                "artificial respondents. Run `python scripts/prepare_reference_data.py --csv survey_results.csv` "
                "and restart the API to use your real survey data."
            )
        else:
            st.caption(f"Computed on {g['reference_rows']} respondents from: {g['reference_source']}")

        gf = pd.DataFrame(
            [{"Feature": f["feature"], "Class": cls, "Mean |SHAP|": v} for f in g["features"] for cls, v in f["per_class"].items()]
        )
        order = [f["feature"] for f in g["features"]]
        by_class = st.toggle("Break down by price range", value=False)
        if by_class:
            ch = alt.Chart(gf).mark_bar().encode(
                y=alt.Y("Feature:N", sort=order, title=None),
                x=alt.X("Mean |SHAP|:Q", stack="zero"),
                color=alt.Color("Class:N", sort=class_labels),
                tooltip=["Feature", "Class", alt.Tooltip("Mean |SHAP|:Q", format=".3f")],
            )
        else:
            overall = pd.DataFrame({"Feature": order, "Mean |SHAP|": [f["mean_abs_shap"] for f in g["features"]]})
            ch = alt.Chart(overall).mark_bar(color="#6366f1").encode(
                y=alt.Y("Feature:N", sort=order, title=None),
                x=alt.X("Mean |SHAP|:Q"),
                tooltip=["Feature", alt.Tooltip("Mean |SHAP|:Q", format=".3f")],
            )
        st.altair_chart(ch.properties(height=460), width="stretch")
        st.caption("Mean absolute SHAP value over respondents and classes: how much each feature moves the model's output on average.")
    except ApiError as exc:
        st.error(str(exc))

# ---- Beeswarm --------------------------------------------------------------
with tab_bee:
    st.markdown("#### SHAP beeswarm (impact and direction)")
    cls = st.selectbox("Price range", class_labels, index=class_labels.index(pred["predicted_price_range"]), key="bee_cls")
    try:
        g = load_global(API_URL, cls)
        if g["is_demo_reference"]:
            st.warning(f"Reference sample: **{g['reference_source']}**.")
        st.pyplot(beeswarm_figure(g["beeswarm"]), clear_figure=True, width="stretch")
        st.caption(
            f"Each dot is one reference respondent; position = SHAP impact on the score for {cls}; colour = the feature's "
            "model-input value (red high, blue low). Colours for encoded categories reflect the encoding, not a natural scale."
        )
    except ApiError as exc:
        st.error(str(exc))
