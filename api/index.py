"""
Value Network API — LightGBM win-probability prediction server

Vercel Serverless Function (Python runtime).
Loads LightGBM models on demand; warm instances reuse cached models.

Supports VR-split models (vote_remaining routing) and 192-feature snapshots.
"""

import os
import math
import hashlib
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
MAX_VR = 5

# {composition: {vr: Booster}}  e.g. {"12B": {0: model, 1: model, ...}}
_models: dict[str, dict[int, lgb.Booster]] = {}
_load_errors: dict[str, str] = {}


def _load_models(composition: str) -> dict[int, lgb.Booster] | None:
    """Load and cache VR-split models for the given composition."""
    if composition in _models:
        return _models[composition]
    if composition in _load_errors:
        return None

    vr_models: dict[int, lgb.Booster] = {}
    try:
        # Try VR-split models first: value_model_{comp}_vr{i}.txt
        for vr in range(MAX_VR + 1):
            path = os.path.join(MODEL_DIR, f"value_model_{composition}_vr{vr}.txt")
            if os.path.exists(path):
                vr_models[vr] = lgb.Booster(model_file=path)

        # Fallback: single model value_model_{comp}.txt → use for all VR
        if not vr_models:
            path = os.path.join(MODEL_DIR, f"value_model_{composition}.txt")
            if os.path.exists(path):
                m = lgb.Booster(model_file=path)
                vr_models = {vr: m for vr in range(MAX_VR + 1)}

        if vr_models:
            _models[composition] = vr_models
            return vr_models
        else:
            _load_errors[composition] = f"No model files found for '{composition}'"
            return None
    except Exception as e:
        _load_errors[composition] = str(e)
        return None


def _get_model(composition: str, vr: int) -> lgb.Booster | None:
    """Get model for composition + vote_remaining, with nearest-neighbor fallback."""
    vr_models = _load_models(composition)
    if not vr_models:
        return None
    if vr in vr_models:
        return vr_models[vr]
    for delta in range(1, MAX_VR + 2):
        if vr - delta in vr_models:
            return vr_models[vr - delta]
        if vr + delta in vr_models:
            return vr_models[vr + delta]
    return None


# Pre-load default model at module level
_load_models("12B")

# ── Encoding maps (must match engine/snapshot.py alphabetical order) ──

ROLE_MAP = {
    "FOX": 0, "GUARD": 1, "MEDIUM": 2, "POSSESSED": 3,
    "SEER": 4, "UNKNOWN": 5, "VILLAGER": 6, "WEREWOLF": 7,
}
CO_MAP = {"GUARD": 0, "MEDIUM": 1, "NONE": 2, "SEER": 3, "FAILED_SEER": 4}
ALIVE_MAP = {"ALIVE": 0, "1DEAD": 1, "2DEAD": 2, "EXECUTED": 3}
OBJECTIVE_STATE_MAP = {
    "CONFIRMED_BLACK": 0, "CONFIRMED_MEDIUM": 1, "CONFIRMED_SEER": 2,
    "CONFIRMED_WHITE": 3, "CONFIRMED_WOLF": 4, "CURSE_CONFIRMED_SEER": 5,
    "FAILED_SEER": 6, "GRAY": 7, "GUARD": 8, "LWCO": 9,
    "MEDIUM": 10, "PANDA": 11, "PARTIALLY_BLACK": 12,
    "PARTIALLY_WHITE": 13, "SEER": 14,
}

# 特徴量数: my_role(1) + fellow_slot(1) + p0(14) + p1~p10(16×10) + dead(6) + agg(8) + vr(1) + rora(1) = 192
SNAPSHOT_N_FEATURES = 2 + 14 + 10 * 16 + 6 + 8 + 1 + 1  # = 192


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
    """Convert a SimSnapshot dict (192 features) to float32 array.

    Layout (matches engine/snapshot.py take_snapshot_numpy):
      [0]       my_role
      [1]       fellow_slot
      [2-15]    p0: co, co_num, rf_0..rf_9, objective_state, wolf_vote_count (14)
      [16-31]   p1: co, co_num, alive, death_day, rf_0..rf_9, objective_state, wolf_vote_count (16)
      ...       p2~p10: same as p1
      [176-181] dead_1..dead_6
      [182-189] 8 aggregate features
      [190]     vote_remaining
      [191]     rora_state
    """
    out = np.empty(SNAPSHOT_N_FEATURES, dtype=np.float32)
    idx = 0

    # my_role
    role_str = snap.get("my_role", "UNKNOWN")
    out[idx] = ROLE_MAP.get(role_str, ROLE_MAP["UNKNOWN"])
    idx += 1

    # fellow_slot
    out[idx] = _safe_num(snap.get("fellow_slot"), -1)
    idx += 1

    # Per-player features
    for i in range(11):
        prefix = f"p{i}_"

        # co (string -> int)
        co_str = snap.get(f"{prefix}co", "NONE")
        out[idx] = CO_MAP.get(co_str, CO_MAP["NONE"])

        # co_num
        out[idx + 1] = _safe_num(snap.get(f"{prefix}co_num"))

        off = 2

        # alive + death_day (p0 excluded: always ALIVE/0)
        if i > 0:
            alive_str = snap.get(f"{prefix}alive", "ALIVE")
            out[idx + off] = ALIVE_MAP.get(alive_str, 0)
            out[idx + off + 1] = _safe_num(snap.get(f"{prefix}death_day"))
            off += 2

        # rf_0 .. rf_9
        for j in range(10):
            out[idx + off + j] = _safe_num(snap.get(f"{prefix}rf_{j}"))
        off += 10

        # objective_state
        obj_str = snap.get(f"{prefix}objective_state", "GRAY")
        out[idx + off] = OBJECTIVE_STATE_MAP.get(obj_str, OBJECTIVE_STATE_MAP["GRAY"])
        off += 1

        # wolf_vote_count
        out[idx + off] = _safe_num(snap.get(f"{prefix}wolf_vote_count"))
        off += 1

        idx += off  # p0: 14, p1~p10: 16

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

    # vote_remaining
    out[idx] = _safe_num(snap.get("vote_remaining"))
    idx += 1

    # rora_state
    out[idx] = _safe_num(snap.get("rora_state"))
    idx += 1

    return out


# ── FastAPI app ──

app = FastAPI(title="Value Network API", version="2.0.0")

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


def _model_hash(composition: str) -> str | None:
    """vr0モデルファイルのmd5先頭8文字を返す。"""
    path = os.path.join(MODEL_DIR, f"value_model_{composition}_vr0.txt")
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


@app.get("/health")
def health(composition: str = Query(default="12B")):
    vr_models = _load_models(composition)
    error = _load_errors.get(composition)
    return {
        "status": "ok" if vr_models is not None else "error",
        "model_loaded": vr_models is not None,
        "composition": composition,
        "n_vr_models": len(vr_models) if vr_models else 0,
        "model_hash": _model_hash(composition),
        "error": error,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    vr_models = _load_models(req.composition)
    if vr_models is None:
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
        # Group snapshots by vote_remaining for VR routing
        vr_groups: dict[int, list[tuple[int, dict]]] = {}
        for i, s in enumerate(req.snapshots):
            vr = int(_safe_num(s.get("vote_remaining"), 0))
            vr_groups.setdefault(vr, []).append((i, s))

        all_probs: list = [None] * len(req.snapshots)

        for vr, items in vr_groups.items():
            model = _get_model(req.composition, vr)
            if model is None:
                continue
            indices = [idx for idx, _ in items]
            features = np.stack([encode_snapshot(s) for _, s in items])
            raw_preds = model.predict(features)

            for j, orig_idx in enumerate(indices):
                row = raw_preds[j]
                if hasattr(row, '__iter__'):
                    all_probs[orig_idx] = [_sanitize_float(float(v)) for v in row]
                else:
                    all_probs[orig_idx] = [_sanitize_float(float(row))]

        # Fill any None entries (model not found for that VR)
        for i in range(len(all_probs)):
            if all_probs[i] is None:
                all_probs[i] = []

        return {"probabilities": all_probs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)},
        )
