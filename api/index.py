"""
Value Network API — LightGBM win-probability prediction server

Vercel Serverless Function (Python runtime).
Loads LightGBM models on demand; warm instances reuse cached models.
"""

import os
import math
import ctypes
from typing import Any

# Preload libgomp for Vercel/Lambda (bundled during build)
_lib_dir = os.path.dirname(__file__)
_gomp_path = os.path.join(_lib_dir, "libgomp.so.1")
if os.path.exists(_gomp_path):
    ctypes.cdll.LoadLibrary(_gomp_path)

import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Model loading (cached across warm invocations) ──

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

_models: dict[str, lgb.Booster] = {}
_load_errors: dict[str, str] = {}


def _load_model(composition: str) -> lgb.Booster | None:
    """Load and cache a model for the given composition."""
    if composition in _models:
        return _models[composition]
    if composition in _load_errors:
        return None

    path = os.path.join(MODEL_DIR, f"value_model_{composition}.txt")
    try:
        booster = lgb.Booster(model_file=path)
        _models[composition] = booster
        return booster
    except Exception as e:
        _load_errors[composition] = str(e)
        return None


# Pre-load default model at module level
_load_model("12B")

# ── Encoding maps (must match engine.py alphabetical order) ──

ROLE_MAP = {
    "FOX": 0, "GUARD": 1, "MEDIUM": 2, "POSSESSED": 3,
    "SEER": 4, "UNKNOWN": 5, "VILLAGER": 6, "WEREWOLF": 7,
}
CO_MAP = {"GUARD": 0, "MEDIUM": 1, "NONE": 2, "SEER": 3}
ALIVE_MAP = {"ALIVE": 0, "ATTACKED": 1, "CURSED": 2, "EXECUTED": 3}

# 特徴量数: 11 players × (5 base + 10 rf) + dead×6 + agg×8 + game_day + vote×33
# = 11 × 15 + 6 + 8 + 1 + 33 = 213  (engine.py take_snapshot_numpy 準拠)
SNAPSHOT_N_FEATURES = 11 * 15 + 6 + 8 + 1 + 33  # = 213


def _safe_num(val: Any, default: float = 0) -> float:
    """Safely convert a value to float, treating None as default."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if not math.isnan(f) else default
    except (TypeError, ValueError):
        return default


def encode_snapshot(snap: dict[str, Any]) -> np.ndarray:
    """Convert a SimSnapshot dict (213 features) to float32 array.

    Layout (matches engine.py take_snapshot_numpy):
      [0-164]   11 players × 15 per-player (role,co,co_num,alive,death_day,rf_0..rf_9)
      [165-170]  dead_1..dead_6
      [171-178]  8 aggregate features
      [179]      game_day_feat
      [180-212]  33 vote target features (11 players × 3 free vote days)
    """
    out = np.empty(SNAPSHOT_N_FEATURES, dtype=np.float32)
    idx = 0

    # Per-player features: 11 players (p0=self, p1..p10=others, NPC excluded)
    for i in range(11):
        prefix = f"p{i}_"

        # role (string -> int)
        role_str = snap.get(f"{prefix}role", "UNKNOWN")
        out[idx] = ROLE_MAP.get(role_str, ROLE_MAP["UNKNOWN"])

        # co (string -> int)
        co_str = snap.get(f"{prefix}co", "NONE")
        out[idx + 1] = CO_MAP.get(co_str, CO_MAP["NONE"])

        # co_num (number)
        out[idx + 2] = _safe_num(snap.get(f"{prefix}co_num"))

        # alive (string -> int)
        alive_str = snap.get(f"{prefix}alive", "ALIVE")
        out[idx + 3] = ALIVE_MAP.get(alive_str, 0)

        # death_day (number | null -> 0 if null)
        out[idx + 4] = _safe_num(snap.get(f"{prefix}death_day"))

        # rf_0 .. rf_9 (10 result features — other sorted players)
        for j in range(10):
            out[idx + 5 + j] = _safe_num(snap.get(f"{prefix}rf_{j}"))

        idx += 15  # 5 base + 10 rf

    # Global: dead_1 .. dead_6
    for d in range(1, 7):
        val = snap.get(f"dead_{d}")
        out[idx] = float(val) if val is not None else np.nan
        idx += 1

    # Aggregate features (8)
    out[idx] = _safe_num(snap.get("agg_seer_co"))
    out[idx + 1] = _safe_num(snap.get("agg_medium_co"))
    out[idx + 2] = _safe_num(snap.get("agg_none_count"))
    out[idx + 3] = _safe_num(snap.get("agg_none_alive"))
    out[idx + 4] = _safe_num(snap.get("agg_none_black"))
    out[idx + 5] = _safe_num(snap.get("agg_none_white"))
    out[idx + 6] = _safe_num(snap.get("agg_none_grey"))
    out[idx + 7] = _safe_num(snap.get("agg_alive_total"))
    idx += 8

    # game_day
    out[idx] = _safe_num(snap.get("game_day_feat"))
    idx += 1

    # Vote target features: vote_0 .. vote_32 (11 players × 3 free vote days)
    for v in range(33):
        val = snap.get(f"vote_{v}")
        out[idx] = float(val) if val is not None else np.nan
        idx += 1

    return out


# ── FastAPI app ──

app = FastAPI(title="Value Network API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    snapshots: list[dict[str, Any]]
    composition: str = "12B"


def _sanitize_float(v: float) -> float | None:
    """Convert NaN/Inf to None for JSON serialization."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@app.get("/health")
def health(composition: str = Query(default="12B")):
    booster = _load_model(composition)
    error = _load_errors.get(composition)
    return {
        "status": "ok" if booster is not None else "error",
        "model_loaded": booster is not None,
        "composition": composition,
        "error": error,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    booster = _load_model(req.composition)
    if booster is None:
        error = _load_errors.get(req.composition)
        return JSONResponse(
            status_code=503,
            content={
                "error": f"Model not loaded for composition '{req.composition}'",
                "detail": error,
            },
        )

    if not req.snapshots:
        return {"probabilities": []}

    try:
        features = np.stack([encode_snapshot(s) for s in req.snapshots])
        raw_preds = booster.predict(features)

        # LightGBM multiclass returns (N, 4) softmax: [P(village), P(wolf), P(fox), P(draw)]
        probs = [[_sanitize_float(float(v)) for v in row] for row in raw_preds]
        return {"probabilities": probs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)},
        )
