"""
Value Network API — LightGBM win-probability prediction server

Vercel Serverless Function (Python runtime).
Loads the LightGBM model once at module level; warm instances reuse it.
"""

import os
import math
from typing import Any

import numpy as np
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Model loading (module-level, cached across warm invocations) ──

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "value_model.txt")

_booster: lgb.Booster | None = None
_load_error: str | None = None

try:
    _booster = lgb.Booster(model_file=MODEL_PATH)
except Exception as e:
    _load_error = str(e)

# ── Encoding maps (must match engine.py alphabetical order) ──

ROLE_MAP = {
    "FOX": 0, "GUARD": 1, "MEDIUM": 2, "POSSESSED": 3,
    "SEER": 4, "UNKNOWN": 5, "VILLAGER": 6, "WEREWOLF": 7,
}
CO_MAP = {"GUARD": 0, "MEDIUM": 1, "NONE": 2, "SEER": 3}
ALIVE_MAP = {"ALIVE": 0, "ATTACKED": 1, "CURSED": 2, "EXECUTED": 3}

SNAPSHOT_N_FEATURES = 12 * 16 + 10  # 202


def encode_snapshot(snap: dict[str, Any]) -> np.ndarray:
    """Convert a SimSnapshot dict (202 string/number fields) to float32 array."""
    out = np.empty(SNAPSHOT_N_FEATURES, dtype=np.float32)

    for i in range(12):
        idx = i * 16
        prefix = f"p{i}_"

        # role (string -> int)
        role_str = snap.get(f"{prefix}role", "UNKNOWN")
        out[idx] = ROLE_MAP.get(role_str, ROLE_MAP["UNKNOWN"])

        # co (string -> int)
        co_str = snap.get(f"{prefix}co", "NONE")
        out[idx + 1] = CO_MAP.get(co_str, CO_MAP["NONE"])

        # co_num (number)
        out[idx + 2] = snap.get(f"{prefix}co_num", 0) or 0

        # alive (string -> int)
        alive_str = snap.get(f"{prefix}alive", "ALIVE")
        out[idx + 3] = ALIVE_MAP.get(alive_str, 0)

        # death_day (number | null -> 0 if null)
        dd = snap.get(f"{prefix}death_day")
        out[idx + 4] = dd if dd is not None else 0

        # rf_0 .. rf_10 (11 result features)
        for j in range(11):
            out[idx + 5 + j] = snap.get(f"{prefix}rf_{j}", 0) or 0

    # Global: dead_1 .. dead_10
    base = 12 * 16
    for d in range(10):
        val = snap.get(f"dead_{d + 1}")
        if val is None:
            out[base + d] = np.nan
        else:
            out[base + d] = val

    return out


# ── FastAPI app ──

app = FastAPI(title="Value Network API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    snapshots: list[dict[str, Any]]


def _sanitize_float(v: float) -> float | None:
    """Convert NaN/Inf to None for JSON serialization."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@app.get("/health")
def health():
    return {
        "status": "ok" if _booster is not None else "error",
        "model_loaded": _booster is not None,
        "error": _load_error,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if _booster is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded", "detail": _load_error},
        )

    if not req.snapshots:
        return {"probabilities": []}

    try:
        features = np.stack([encode_snapshot(s) for s in req.snapshots])
        raw_preds = _booster.predict(features)

        # LightGBM binary classifier returns P(village_win)
        probs = [_sanitize_float(float(p)) for p in raw_preds]
        return {"probabilities": probs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)},
        )
