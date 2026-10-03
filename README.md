# Explainable Machine Learning for Consumer Price Preference Modeling

An end-to-end Explainable AI (XAI) application for predicting a
consumer's preferred product price range from demographic, consumption,
brand, purchase, packaging, and health-related attributes.

The project combines **XGBoost**, **SHAP**, **FastAPI**, and
**Streamlit** to provide predictions together with model explanations.

## Key Features

-   XGBoost price-range classification
-   Probability distribution for all price classes
-   Automatic feature engineering from raw inputs
-   Dynamic local SHAP explanations for the current user
-   Global SHAP feature importance and beeswarm analysis
-   FastAPI inference backend
-   Streamlit interactive frontend
-   Health-check and model-information endpoints

> SHAP describes how features contributed to the model prediction. It
> does not establish causal relationships.

## Model

Final serialized pipeline:

``` text
artifacts/xgboost_final_model.joblib
```

Reported model evaluation:

-   Cross-validation accuracy: **0.926692**
-   Test accuracy: **0.927093**
-   Macro F1: **0.92**
-   Train/test split: **75/25**
-   Random state: **42**
-   Rows after cleaning: **29,956**
-   Test rows: **7,489**

Final XGBoost configuration includes:

``` text
n_estimators = 300
max_depth = 5
learning_rate = 0.2
subsample = 0.8
min_child_weight = 7
gamma = 0.5
colsample_bytree = 0.7
```

## Price Classes

``` text
50-100
100-150
150-200
200-250
```

## Raw Input Features

``` text
respondent_id
age
gender
zone
occupation
income_levels
consume_frequency(weekly)
current_brand
preferable_consumption_size
awareness_of_other_brands
reasons_for_choosing_brands
flavor_preference
purchase_channel
packaging_preference
health_concerns
typical_consumption_situations
```

Target:

``` text
price_range
```

## Feature Engineering

The API recreates the training feature-engineering logic automatically.

### Age Group

``` text
18-25, 26-35, 36-45, 46-55, 56-70, 70+
```

### Frequency Score

``` text
0-2 times -> 1
3-4 times -> 2
5-7 times -> 3
```

### Brand Awareness Score

``` text
0 to 1 -> 1
2 to 4 -> 2
above 4 -> 3
```

### CFAB

``` text
CFAB = frequency_score / (awareness_score + frequency_score)
```

Rounded to two decimal places.

### Zone Score

``` text
Urban -> 3
Metro -> 4
Semi-Urban -> 2
Rural -> 1
```

### Income Score

``` text
<10L -> 1
10L - 15L -> 2
16L - 25L -> 3
26L - 35L -> 4
> 35L -> 5
Not Reported -> 0
```

### ZAS

``` text
ZAS = zone_score * income_score
```

### BSI

BSI is `1` when the current brand is not `Established` and the reason
for choosing the brand is `Price` or `Quality`; otherwise it is `0`.

## Explainable AI

The application uses SHAP `TreeExplainer` with the XGBoost model.

### Local SHAP

For the current user's input, the application returns:

-   predicted price range
-   explained class
-   explained-class probability
-   baseline output
-   final model output
-   positive contributors
-   negative contributors
-   feature contributions
-   additivity error
-   human-readable explanation

The explanation is generated dynamically for the current input rather
than using a fixed sample.

### Global SHAP

The application also supports global feature importance and SHAP
beeswarm analysis.

Reference data:

``` text
artifacts/reference_data.csv
```

## Architecture

``` text
User
  |
  v
Streamlit Frontend
  |
  | HTTP
  v
FastAPI Backend
  |
  +--> Feature Engineering
  |
  +--> XGBoost
  |
  +--> Prediction / Probabilities
  |
  +--> SHAP
  |
  v
Explanation Response
  |
  v
Streamlit UI
```

## Project Structure

``` text
project/
├── app/
│   ├── __init__.py
│   ├── explainer.py
│   ├── main.py
│   ├── predictor.py
│   ├── routes.py
│   └── schemas.py
│
├── artifacts/
│   ├── label_mapping.json
│   ├── model_metadata.json
│   ├── reference_data.csv
│   ├── reference_meta.json
│   └── xgboost_final_model.joblib
│
├── scripts/
│   └── prepare_reference_data.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
├── pytest.ini
├── README.md
├── requirements.txt
└── run.py
```

## API Endpoints

  Method   Endpoint                Purpose
  -------- ----------------------- --------------------------------
  GET      `/health`               Backend/model health
  POST     `/predict`              Prediction and probabilities
  POST     `/explain`              Dynamic local SHAP explanation
  GET      `/global-explanation`   Global SHAP analysis
  GET      `/model-info`           Model and environment metadata

Swagger documentation:

``` text
http://127.0.0.1:8000/docs
```

## Local Setup

Create and activate the virtual environment:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install -r requirements.txt
```

Start FastAPI:

``` powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal:

``` powershell
python -m streamlit run .\streamlit_apppp.py
```

Open:

``` text
http://localhost:8501
```

## Compatibility

The serialized pipeline requires the compatible ML environment. In
particular:

``` text
scikit-learn==1.6.1
```

Do not casually upgrade scikit-learn and then load the serialized model.

## Testing

Run:

``` powershell
pytest -q
```

Recommended manual API sequence:

``` text
/health
/model-info
/predict
/explain
/global-explanation
```

## GitHub

Before pushing, do not commit:

``` text
.venv/
__pycache__/
*.pyc
.env
```

Use a `.gitignore` file.

## Streamlit Community Cloud Deployment

The Streamlit frontend can be deployed from GitHub through Streamlit
Community Cloud.

Important: this project currently uses a **Streamlit -\> FastAPI**
architecture. A deployed Streamlit app cannot call:

``` text
http://127.0.0.1:8000
```

because that address refers to the remote Streamlit server itself.

Therefore, for cloud deployment, the FastAPI backend must also be
deployed to a publicly accessible HTTPS URL. The Streamlit app should
use that backend URL through an environment variable or Streamlit secret
rather than hard-coding localhost.

Deployment architecture:

``` text
Streamlit Community Cloud
        |
        | HTTPS
        v
Public FastAPI Backend
        |
        v
XGBoost + SHAP
```

## Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   SHAP
-   Matplotlib
-   Plotly
-   FastAPI
-   Uvicorn
-   Pydantic
-   Streamlit
-   Joblib
-   Pytest
-   HTTPX

## Author

**Tanmay Pakori**

MSc Mathematics & Data Science

GitHub: `MrTanmay18`

## License

This project is intended for educational, portfolio, and demonstration
purposes.
