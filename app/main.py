"""FastAPI entry point.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
The model, preprocessing pipeline and SHAP explainer are loaded ONCE at start-up.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.explainer import ShapService
from app.predictor import ModelService
from app.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("price_intel.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model and SHAP explainer ...")
    model_service = ModelService()
    shap_service = ShapService(model_service)
    shap_service.warm_up()  # global SHAP; non-fatal if no reference data
    app.state.model_service = model_service
    app.state.shap_service = shap_service
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Consumer Price Preference Intelligence API",
    description="XGBoost price-range prediction with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
