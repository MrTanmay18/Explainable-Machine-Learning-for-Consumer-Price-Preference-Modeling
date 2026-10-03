"""HTTP endpoints. Services are created once in ``main.py`` and read from ``app.state``."""
from __future__ import annotations

from typing import Optional

import shap
from fastapi import APIRouter, HTTPException, Query, Request

from app.explainer import ShapService
from app.predictor import ModelService
from app.schemas import (
    ConsumerInput,
    ExplainResponse,
    GlobalExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

router = APIRouter()


def _services(request: Request) -> tuple[ModelService, ShapService]:
    ms = getattr(request.app.state, "model_service", None)
    ex = getattr(request.app.state, "shap_service", None)
    if ms is None or ex is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return ms, ex


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    ms = getattr(request.app.state, "model_service", None)
    ex = getattr(request.app.state, "shap_service", None)
    return HealthResponse(
        status="ok" if ms is not None and ex is not None else "degraded",
        model_loaded=ms is not None,
        explainer_ready=ex is not None,
        reference_data_ready=bool(ex and ex.reference_info.get("available")),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: ConsumerInput, request: Request) -> PredictionResponse:
    """Predict the price range from the 15 raw survey inputs (features are engineered server-side)."""
    ms, _ = _services(request)
    try:
        out = ms.predict(payload.to_raw_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out.pop("_frame")
    return PredictionResponse(**out)


@router.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(
    payload: ConsumerInput,
    request: Request,
    target_class: Optional[str] = Query(
        None, description="Price range to explain, e.g. '100-150'. Defaults to the predicted class."
    ),
) -> ExplainResponse:
    """Local SHAP explanation for THIS input (baseline, contributions, final output, narrative)."""
    ms, ex = _services(request)
    try:
        frame = ms.build_frame(payload.to_raw_dict())
        return ExplainResponse(**ex.explain_local(frame, target_class))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/global-explanation", response_model=GlobalExplanationResponse, tags=["inference"])
def global_explanation(
    request: Request,
    target_class: Optional[str] = Query(None, description="Price range for the beeswarm plot."),
    top_k: int = Query(15, ge=3, le=31),
    max_points: int = Query(800, ge=50, le=2000),
) -> GlobalExplanationResponse:
    """Global feature importance + beeswarm data computed from the reference sample."""
    _, ex = _services(request)
    try:
        return GlobalExplanationResponse(**ex.explain_global(target_class, top_k, max_points))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info(request: Request) -> ModelInfoResponse:
    ms, ex = _services(request)
    info = ms.info()
    info["library_versions"]["shap"] = shap.__version__
    info["reference_data"] = {
        "available": ex.reference_info["available"],
        "source": ex.reference_info["source"],
        "rows": ex.reference_info["rows"],
        "is_demo": ex.reference_info["is_demo"],
    }
    return ModelInfoResponse(**info)
