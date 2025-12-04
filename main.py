from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import io
import os
import threading
import numpy as np
from typing import Dict, List, Any, Optional
import math
import difflib
import re
from fastapi.middleware.cors import CORSMiddleware
from parse_v2 import parse_v2_services, parse_v2_products

# FastAPI app
app = FastAPI(title="Inventory Data Parser", description="AI-powered inventory data normalization service")

# CORS for browser clients
allowed_origins_env = os.environ.get("CORS_ORIGINS")
allowed_origins = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else [
    'http://localhost:5174',
    'http://localhost:5173',
    'http://localhost:3000',
    'https://ycc-sage.vercel.app',
    'https://ycc-client.vercel.app',
    'https://ycc-client.netlify.app',
    'https://yachtcrewcenter-dev.netlify.app',
    'https://yachtcrewcenter.com',
    'https://staging.yachtcrewcenter.com',
    'https://staging.yachtcrewcenter.com/'
]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Hugging Face cache to ephemeral disk (safe for Heroku dynos)
os.environ.setdefault("HF_HOME", "/tmp/hf")

# === Target fields expected by Node ===
TARGET_FIELDS: List[str] = [
    "name",
    "category",
    "description",
    "price",
    "sku",
    "hsCode",
    "countryOfOrigin",
    "weight",
    "height",
    "length",
    "width",
    "quantity",
]

# Required fields to accept the file as valid (sku and description are optional)
REQUIRED_FIELDS: List[str] = [
    "name",
    "price",
    "hsCode",
    "countryOfOrigin",
    "weight",
    "height",
    "length",
    "width",
    "quantity",
]

# === Service-specific target fields ===
SERVICE_TARGET_FIELDS: List[str] = [
    "serviceName",
    "serviceCategory", 
    "description",
    "price",
]

# Required fields for services
SERVICE_REQUIRED_FIELDS: List[str] = [
    "serviceName",
    "serviceCategory",
    "price",
]

# === Synonyms for column name mapping ===
FIELD_SYNONYMS: Dict[str, List[str]] = {
    "name": [
        "name",
        "product name",
        "product",
        "item",
        "item name",
        "title",
        "product title",
        "product label",
        "item title",
        "product description title",
    ],
    "category": [
        "category",
        "categories",
        "group",
        "type",
        "department",
        "section",
        "collection",
    ],
    "description": [
        "description",
        "details",
        "info",
        "information",
        "specs",
        "specifications",
        "product details",
    ],
    "price": [
        "price",
        "unit price",
        "amount",
        "cost",
        "value",
        "rate",
        "selling price",
        "list price",
    ],
    "sku": [
        "sku",
        "item code",
        "product code",
        "code",
        "id",
        "identifier",
        "sku code",
        "stock keeping unit",
    ],
    "hsCode": [
        "hs code",
        "hscode",
        "hs",
        "tariff code",
        "customs code",
        "harmonized code",
        "hs number",
    ],
    "countryOfOrigin": [
        "country of origin",
        "origin",
        "country",
        "made in",
        "manufactured in",
        "origin country",
        "coo",
    ],
    "weight": ["weight", "wt", "mass", "net weight", "gross weight"],
    "height": ["height", "h"],
    "length": ["length", "l"],
    "width": ["width", "w"],
    "quantity": ["quantity", "qty", "stock", "inventory", "available", "qtty"],
}

# === Service-specific synonyms for column name mapping ===
SERVICE_FIELD_SYNONYMS: Dict[str, List[str]] = {
    "serviceName": [
        "service name",
        "service",
        "name",
        "service title",
        "title",
        "service description",
        "service type",
        "offering",
        "service offering",
    ],
    "serviceCategory": [
        "service category",
        "category",
        "service type",
        "type",
        "group",
        "classification",
        "department",
        "service group",
        "service class",
    ],
    "description": [
        "description",
        "details",
        "info",
        "information",
        "service details",
        "service description",
        "overview",
        "summary",
        "about",
    ],
    "price": [
        "price",
        "cost",
        "rate",
        "fee",
        "charge",
        "amount",
        "service price",
        "service cost",
        "service rate",
        "service fee",
        "hourly rate",
        "daily rate",
    ],
}

# Nicer display labels for fields
FIELD_LABELS: Dict[str, str] = {
    "name": "Name",
    "category": "Category",
    "description": "Description",
    "price": "Price",
    "sku": "SKU",
    "hsCode": "HS Code",
    "countryOfOrigin": "Country of Origin",
    "weight": "Weight",
    "height": "Height",
    "length": "Length",
    "width": "Width",
    "quantity": "Quantity",
}

# Service-specific display labels
SERVICE_FIELD_LABELS: Dict[str, str] = {
    "serviceName": "Service Name",
    "serviceCategory": "Service Category", 
    "description": "Description",
    "price": "Price",
}

# === Lazy-loaded embedding model ===
_model = None
_model_lock = threading.Lock()
_model_loading = False


def _load_model_if_needed(non_blocking: bool = False) -> bool:
    """Ensure the SentenceTransformer model is loaded.
    Returns True if loaded, False if a background load has started (or is in progress).
    """
    global _model, _model_loading
    if _model is not None:
        return True

    if non_blocking:
        # Start background load if not already in progress
        with _model_lock:
            if _model is None and not _model_loading:
                _model_loading = True
                threading.Thread(target=_blocking_load_model, daemon=True).start()
        return False

    # Blocking load
    _blocking_load_model()
    return _model is not None


def _blocking_load_model():
    global _model, _model_loading
    with _model_lock:
        if _model is not None:
            return
        try:
            _model_loading = True
            # Small, fast model suitable for CPU-only dynos
            _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        finally:
            _model_loading = False


# Kick off background warm-up after startup (does not block port binding)
@app.on_event("startup")
async def warm_up_model():
    _load_model_if_needed(non_blocking=True)


# === Utility: safe numeric coercion (tolerates units like '5L', '2.5kg') ===
def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            f = float(value)
            return f if math.isfinite(f) else None
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if s == "":
                return None
            # Extract first numeric token (optional sign, digits, optional decimal)
            import re
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            if not m:
                return None
            f = float(m.group(0))
            return f if math.isfinite(f) else None
        # Fallback: try casting generically
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    try:
        i = int(round(f))
        return i
    except Exception:
        return None


# === Column mapping via synonyms + semantic fallback ===
def _normalize_header(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s_\-]+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _normalize_header(text).split(" ") if tok]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _string_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_header(a), _normalize_header(b)).ratio()


def _column_features(series: pd.Series) -> Dict[str, Any]:
    n = int(series.shape[0])
    sample = series.dropna().astype(str).head(200).tolist()
    if n == 0:
        return {
            "p_numeric": 0.0,
            "p_integer": 0.0,
            "avg_len": 0.0,
            "hs_like": 0.0,
            "has_weight_units": 0.0,
            "low_cardinality": False,
        }
    numeric_flags = []
    integer_flags = []
    lengths = []
    hs_hits = 0
    unit_hits = 0
    for v in sample:
        s = str(v).strip()
        lengths.append(len(s))
        # numeric
        try:
            float_val = float(re.sub(r"[^0-9.\-]", "", s)) if re.search(r"\d", s) else None
        except Exception:
            float_val = None
        numeric_flags.append(1 if float_val is not None else 0)
        if float_val is not None:
            integer_flags.append(1 if abs(float_val - round(float_val)) < 1e-9 else 0)
        else:
            integer_flags.append(0)
        # hs-like: 4-10 consecutive digits possibly with spaces/hyphens
        if re.search(r"\b\d{4,10}\b", s):
            hs_hits += 1
        # weight units
        if re.search(r"\b(kg|g|gram|grams|lb|lbs|pound|oz|ounce|ounces)\b", s, re.I):
            unit_hits += 1
    unique_ratio = len(set(sample)) / max(1, len(sample))
    return {
        "p_numeric": sum(numeric_flags) / max(1, len(sample)),
        "p_integer": sum(integer_flags) / max(1, len(sample)),
        "avg_len": sum(lengths) / max(1, len(lengths)),
        "hs_like": hs_hits / max(1, len(sample)),
        "has_weight_units": unit_hits / max(1, len(sample)),
        "low_cardinality": unique_ratio < 0.5,
    }


def _service_heuristic_boost(field: str, features: Dict[str, Any]) -> float:
    """Service-specific heuristic scoring boost based on column content analysis."""
    boost = 0.0
    if field == "price":
        if features["p_numeric"] > 0.8 and features["p_integer"] < 0.9:
            boost += 0.25
    if field == "serviceCategory":
        if features["low_cardinality"] and features["p_numeric"] < 0.3:
            boost += 0.2
    if field == "serviceName":
        if features["p_numeric"] < 0.2 and features["avg_len"] >= 3:
            boost += 0.25
    if field == "description":
        if features["p_numeric"] < 0.1 and features["avg_len"] >= 10:
            boost += 0.15
    return boost


def _heuristic_boost(field: str, features: Dict[str, Any]) -> float:
    boost = 0.0
    if field == "price":
        if features["p_numeric"] > 0.8 and features["p_integer"] < 0.9:
            boost += 0.2
    if field == "quantity":
        if features["p_numeric"] > 0.8 and features["p_integer"] > 0.8:
            boost += 0.25
    if field in {"weight", "height", "length", "width"}:
        if features["p_numeric"] > 0.75:
            boost += 0.2
        if field == "weight" and features["has_weight_units"] > 0.05:
            boost += 0.2
    if field == "hsCode":
        if features["hs_like"] > 0.1 and features["p_numeric"] > 0.5:
            boost += 0.3
    if field == "category":
        if features["low_cardinality"] and features["p_numeric"] < 0.3:
            boost += 0.15
    if field == "name":
        if features["p_numeric"] < 0.2 and features["avg_len"] >= 3:
            boost += 0.2
    return boost


def _build_service_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Build column mapping for service fields using synonyms and semantic similarity."""
    user_columns = list(df.columns)
    normalized_to_original = { _normalize_header(c): c for c in user_columns }
    normalized_user_headers = list(normalized_to_original.keys())

    column_map: Dict[str, Optional[str]] = {field: None for field in SERVICE_TARGET_FIELDS}

    # Pass 1: exact/synonym match on normalized headers
    normalized_synonyms: Dict[str, List[str]] = {
        f: [_normalize_header(s) for s in syns] for f, syns in SERVICE_FIELD_SYNONYMS.items()
    }
    for field, syns in normalized_synonyms.items():
        for h_norm in normalized_user_headers:
            if h_norm in syns:
                column_map[field] = normalized_to_original[h_norm]
                break

    # Prepare candidates not yet used
    unmapped_fields = [f for f, c in column_map.items() if c is None]
    candidate_columns = [c for c in user_columns if c not in column_map.values()]

    # Precompute features for candidates
    col_features: Dict[str, Dict[str, Any]] = {
        col: _column_features(df[col]) for col in candidate_columns
    }

    # Precompute tokens of field labels
    field_label_tokens = { f: _tokenize(SERVICE_FIELD_LABELS.get(f, f)) for f in unmapped_fields }

    # Pass 2: fuzzy token/string similarity
    scores: List[tuple] = []  # (score, field, col)
    for field in unmapped_fields:
        for col in candidate_columns:
            h = col
            # token overlap vs the best of field label or any synonym
            tokens_h = _tokenize(h)
            best_tok = max([
                _jaccard(tokens_h, _tokenize(p))
                for p in [SERVICE_FIELD_LABELS.get(field, field)] + SERVICE_FIELD_SYNONYMS.get(field, [])
            ] or [0.0])
            best_str = max([
                _string_sim(h, p)
                for p in [SERVICE_FIELD_LABELS.get(field, field)] + SERVICE_FIELD_SYNONYMS.get(field, [])
            ] or [0.0])
            base = 0.35 * best_tok + 0.35 * best_str
            base += _service_heuristic_boost(field, col_features[col])
            if base > 0:
                scores.append((base, field, col))

    # Pass 3: semantic similarity with header-only weighting
    if unmapped_fields and candidate_columns:
        if _load_model_if_needed(non_blocking=False):
            # Encode candidate headers
            header_texts = [ _normalize_header(c) for c in candidate_columns ]
            cand_emb = _model.encode(header_texts, convert_to_numpy=True)
            cand_norm = np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-12
            cand_n = cand_emb / cand_norm

            for field in unmapped_fields:
                phrases = [SERVICE_FIELD_LABELS.get(field, field)] + SERVICE_FIELD_SYNONYMS.get(field, [])
                field_emb = _model.encode([_normalize_header(p) for p in phrases], convert_to_numpy=True)
                field_norm = np.linalg.norm(field_emb, axis=1, keepdims=True) + 1e-12
                field_n = field_emb / field_norm
                # Use the best phrase score for the field
                # cosine matrix: candidates x phrases
                cos = cand_n @ field_n.T
                best_per_cand = cos.max(axis=1)
                for idx, col in enumerate(candidate_columns):
                    sem = float(best_per_cand[idx])
                    if sem > 0:
                        score = 0.8 * sem + _service_heuristic_boost(field, col_features[col])
                        scores.append((score, field, col))

            # Free embeddings
            del cand_emb, cand_norm, cand_n

    # Aggregate and choose assignments greedily by highest score with threshold
    # Sum scores across passes
    from collections import defaultdict
    agg: Dict[tuple, float] = defaultdict(float)
    for s, f, c in scores:
        agg[(f, c)] += s

    assignments = []
    used_fields = set()
    used_cols = set()
    # Sort by score descending
    for (f, c), s in sorted(agg.items(), key=lambda x: x[1], reverse=True):
        # Minimum acceptance threshold
        if s < 0.85:
            continue
        if f in used_fields or c in used_cols:
            continue
        assignments.append((f, c, s))
        used_fields.add(f)
        used_cols.add(c)

    for f, c, _ in assignments:
        column_map[f] = c

    return column_map


def _build_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    user_columns = list(df.columns)
    normalized_to_original = { _normalize_header(c): c for c in user_columns }
    normalized_user_headers = list(normalized_to_original.keys())

    column_map: Dict[str, Optional[str]] = {field: None for field in TARGET_FIELDS}

    # Pass 1: exact/synonym match on normalized headers
    normalized_synonyms: Dict[str, List[str]] = {
        f: [_normalize_header(s) for s in syns] for f, syns in FIELD_SYNONYMS.items()
    }
    for field, syns in normalized_synonyms.items():
        for h_norm in normalized_user_headers:
            if h_norm in syns:
                column_map[field] = normalized_to_original[h_norm]
                break

    # Prepare candidates not yet used
    unmapped_fields = [f for f, c in column_map.items() if c is None]
    candidate_columns = [c for c in user_columns if c not in column_map.values()]

    # Precompute features for candidates
    col_features: Dict[str, Dict[str, Any]] = {
        col: _column_features(df[col]) for col in candidate_columns
    }

    # Precompute tokens of field labels
    field_label_tokens = { f: _tokenize(FIELD_LABELS.get(f, f)) for f in unmapped_fields }

    # Pass 2: fuzzy token/string similarity
    scores: List[tuple] = []  # (score, field, col)
    for field in unmapped_fields:
        for col in candidate_columns:
            h = col
            # token overlap vs the best of field label or any synonym
            tokens_h = _tokenize(h)
            best_tok = max([
                _jaccard(tokens_h, _tokenize(p))
                for p in [FIELD_LABELS.get(field, field)] + FIELD_SYNONYMS.get(field, [])
            ] or [0.0])
            best_str = max([
                _string_sim(h, p)
                for p in [FIELD_LABELS.get(field, field)] + FIELD_SYNONYMS.get(field, [])
            ] or [0.0])
            base = 0.35 * best_tok + 0.35 * best_str
            base += _heuristic_boost(field, col_features[col])
            if base > 0:
                scores.append((base, field, col))

    # Pass 3: semantic similarity with header-only weighting
    if unmapped_fields and candidate_columns:
        if _load_model_if_needed(non_blocking=False):
            # Encode candidate headers
            header_texts = [ _normalize_header(c) for c in candidate_columns ]
            cand_emb = _model.encode(header_texts, convert_to_numpy=True)
            cand_norm = np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-12
            cand_n = cand_emb / cand_norm

            for field in unmapped_fields:
                phrases = [FIELD_LABELS.get(field, field)] + FIELD_SYNONYMS.get(field, [])
                field_emb = _model.encode([_normalize_header(p) for p in phrases], convert_to_numpy=True)
                field_norm = np.linalg.norm(field_emb, axis=1, keepdims=True) + 1e-12
                field_n = field_emb / field_norm
                # Use the best phrase score for the field
                # cosine matrix: candidates x phrases
                cos = cand_n @ field_n.T
                best_per_cand = cos.max(axis=1)
                for idx, col in enumerate(candidate_columns):
                    sem = float(best_per_cand[idx])
                    if sem > 0:
                        score = 0.8 * sem + _heuristic_boost(field, col_features[col])
                        scores.append((score, field, col))

            # Free embeddings
            del cand_emb, cand_norm, cand_n

    # Aggregate and choose assignments greedily by highest score with threshold
    # Sum scores across passes
    from collections import defaultdict
    agg: Dict[tuple, float] = defaultdict(float)
    for s, f, c in scores:
        agg[(f, c)] += s

    assignments = []
    used_fields = set()
    used_cols = set()
    # Sort by score descending
    for (f, c), s in sorted(agg.items(), key=lambda x: x[1], reverse=True):
        # Minimum acceptance threshold
        if s < 0.85:
            continue
        if f in used_fields or c in used_cols:
            continue
        assignments.append((f, c, s))
        used_fields.add(f)
        used_cols.add(c)

    for f, c, _ in assignments:
        column_map[f] = c

    return column_map


def _normalize_category(value: Any) -> List[str]:
    if pd.isna(value) or (isinstance(value, str) and value.strip().lower() in ('', 'nan', 'none')):
        return ["Other"]
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip() and str(v).strip().lower() != 'nan']
        return out or ["Other"]
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return ["Other"]
    parts = [p.strip() for p in re_split(text) if p.strip() and p.strip().lower() != 'nan']
    return parts or ["Other"]


def re_split(text: str) -> List[str]:
    # Helper separated to avoid importing re at module import (keeps startup fast)
    import re
    return re.split(r"[,;|]", text)


def _service_row_errors(row_idx: int, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a service row and return list of errors."""
    errs: List[Dict[str, Any]] = []

    def add(field: str, msg: str):
        errs.append({"row": row_idx, "field": field, "message": msg})

    # Required presence (only for SERVICE_REQUIRED_FIELDS)
    for field in SERVICE_REQUIRED_FIELDS:
        if row.get(field) is None:
            add(field, "required")

    # Type/constraints (only if value present)
    if row.get("price") is not None and (not isinstance(row["price"], (int, float)) or row["price"] < 0):
        add("price", "must be a non-negative number")

    # Service name must be non-empty string
    if row.get("serviceName") is not None and (not isinstance(row["serviceName"], str) or not row["serviceName"].strip()):
        add("serviceName", "must be a non-empty text")

    # Service category must be non-empty list
    if not isinstance(row.get("serviceCategory"), list) or len(row["serviceCategory"]) == 0:
        add("serviceCategory", "must be a non-empty list")

    return errs


def _row_errors(row_idx: int, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []

    def add(field: str, msg: str):
        errs.append({"row": row_idx, "field": field, "message": msg})

    # Required presence (only for REQUIRED_FIELDS)
    for field in REQUIRED_FIELDS:
        if row.get(field) is None:
            add(field, "required")

    # Type/constraints (same as before, but only if value present)
    if row.get("price") is not None and (not isinstance(row["price"], (int, float)) or row["price"] < 0):
        add("price", "must be a non-negative number")
    for dim in ["weight", "height", "length", "width"]:
        if row.get(dim) is not None and (not isinstance(row[dim], (int, float)) or row[dim] < 0):
            add(dim, "must be a non-negative number")
    if row.get("quantity") is not None and (not isinstance(row["quantity"], int) or row["quantity"] < 0):
        add("quantity", "must be a non-negative integer")

    # Category must be non-empty array (but since we default to ["Other"], it shouldn't trigger)
    if not isinstance(row.get("category"), list) or len(row["category"]) == 0:
        add("category", "must be a non-empty list")

    return errs


def _service_friendly_missing_columns_message(missing_fields: List[str]) -> Dict[str, Any]:
    """Generate friendly error message for missing service columns."""
    bullets = []
    for f in missing_fields:
        label = SERVICE_FIELD_LABELS.get(f, f)
        syns = SERVICE_FIELD_SYNONYMS.get(f, [])
        syns_preview = ", ".join(syns[:4]) if syns else None
        if syns_preview:
            bullets.append(f"- {label}: try one of → {syns_preview}")
        else:
            bullets.append(f"- {label}")

    title = "I couldn't find some required service columns."
    hint = (
        "Tip: Use clear headers. You can rename your columns to match the suggested examples above, "
        "then re-upload."
    )
    return {
        "title": title,
        "bullets": bullets,
        "hint": hint,
        "message": title,
    }


def _service_friendly_row_errors_message(errors: List[Dict[str, Any]], column_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Generate friendly error message for service row validation errors."""
    # Group by field + message
    from collections import defaultdict

    grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for err in errors:
        row = err.get("row")
        field = err.get("field")
        msg = err.get("message")
        grouped[field][msg].append(row)

    bullets = []
    for field, msgs in grouped.items():
        original_field_name = column_map.get(field)
        if original_field_name:
            label = original_field_name
        else:
            label = SERVICE_FIELD_LABELS.get(field, field)
        for msg, rows in msgs.items():
            sample = ", ".join(str(r) for r in rows[:5])
            count = len(rows)
            if msg == "required":
                human = f"{label} is missing on {count} row(s) (e.g., row {sample})."
            else:
                human = f"{label} {msg} on {count} row(s) (e.g., row {sample})."
            bullets.append(f"- {human}")

    title = "Some service rows need attention before I can import."
    hint = "Tip: Fix the highlighted issues and re-upload. I'll validate again instantly."
    return {
        "title": title,
        "bullets": bullets,
        "hint": hint,
        "message": title,
    }


def _friendly_missing_columns_message(missing_fields: List[str]) -> Dict[str, Any]:
    bullets = []
    for f in missing_fields:
        label = FIELD_LABELS.get(f, f)
        syns = FIELD_SYNONYMS.get(f, [])
        syns_preview = ", ".join(syns[:4]) if syns else None
        if syns_preview:
            bullets.append(f"- {label}: try one of → {syns_preview}")
        else:
            bullets.append(f"- {label}")

    title = "I couldn't find some required columns."
    hint = (
        "Tip: Use clear headers. You can rename your columns to match the suggested examples above, "
        "then re-upload."
    )
    # Do not inject bullets into message to avoid duplicate rendering client-side
    return {
        "title": title,
        "bullets": bullets,
        "hint": hint,
        "message": title,
    }


def _friendly_row_errors_message(errors: List[Dict[str, Any]], column_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    # Group by field + message
    from collections import defaultdict

    grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for err in errors:
        row = err.get("row")
        field = err.get("field")
        msg = err.get("message")
        original_field_name = column_map.get(field)
        if original_field_name:
            label = original_field_name
        else:
            label = FIELD_LABELS.get(field, field)
        grouped[field][msg].append(row)

    bullets = []
    for field, msgs in grouped.items():
        original_field_name = column_map.get(field)
        if original_field_name:
            label = original_field_name
        else:
            label = FIELD_LABELS.get(field, field)
        for msg, rows in msgs.items():
            sample = ", ".join(str(r) for r in rows[:5])
            count = len(rows)
            if msg == "required":
                human = f"{label} is missing on {count} row(s) (e.g., row {sample})."
            else:
                human = f"{label} {msg} on {count} row(s) (e.g., row {sample})."
            bullets.append(f"- {human}")

    title = "Some rows need attention before I can import."
    hint = "Tip: Fix the highlighted issues and re-upload. I'll validate again instantly."
    return {
        "title": title,
        "bullets": bullets,
        "hint": hint,
        "message": title,
    }


@app.get("/")
async def root():
    return {"message": "Inventory Data Parser API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _model is not None, "model_loading": _model_loading}


@app.post("/parse-inventory")
async def parse_inventory(file: UploadFile = File(...)):
    """
    Parse and normalize inventory data from uploaded Excel file.
    Returns 200 with products[] ready for Node import, or 422 with row-level errors.
    """
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith((".xlsx", ".xls")):
            return JSONResponse(status_code=400, content={"message": "Please upload an Excel file (.xlsx or .xls)."})

        content = await file.read()
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception:
            return JSONResponse(status_code=400, content={"message": "We couldn't read that spreadsheet. Please check the file format and try again."})

        # Drop fully empty rows/columns
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        if df.shape[0] == 0:
            friendly = {
                "title": "I couldn't find any data rows.",
                "bullets": ["Make sure your sheet has a header row and at least one data row."],
                "message": "I couldn't find any data rows. Make sure your sheet has a header row and at least one data row.",
            }
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": [{"row": None, "field": "file", "message": "No data rows found"}],
                "friendly": friendly,
            })

        # Build column map
        column_map = _build_column_map(df)

        # Validate required mappings exist
        missing_required_cols = [f for f in REQUIRED_FIELDS if column_map.get(f) is None]
        if missing_required_cols:
            friendly = _friendly_missing_columns_message(missing_required_cols)
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": [
                    {"row": None, "field": f, "message": "required column not found"}
                    for f in missing_required_cols
                ],
                "friendly": friendly,
            })

        # Build normalized products
        products: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx in range(len(df)):
            row_src = df.iloc[idx]

            def get_val(field: str) -> Any:
                col = column_map.get(field)
                if not col:
                    return None
                try:
                    val = row_src[col]
                    if pd.isna(val) or (isinstance(val, str) and val.strip().lower() in ('', 'nan', 'none')):
                        return None
                    return val
                except Exception:
                    return None

            # Extract and normalize
            name = str(get_val("name")).strip() if get_val("name") is not None else None
            category_raw = get_val("category")
            category = _normalize_category(category_raw)
            description_raw = get_val("description")
            description = None if description_raw is None or str(description_raw).strip() == "" else str(description_raw).strip()
            price = _to_float(get_val("price"))
            sku_raw = get_val("sku")
            sku = None if sku_raw is None or str(sku_raw).strip() == "" else str(sku_raw).strip()
            hs_code = None if get_val("hsCode") is None else str(get_val("hsCode")).strip()
            country = None if get_val("countryOfOrigin") is None else str(get_val("countryOfOrigin")).strip()
            weight = _to_float(get_val("weight"))
            height = _to_float(get_val("height"))
            length = _to_float(get_val("length"))
            width = _to_float(get_val("width"))
            quantity = _to_int(get_val("quantity"))

            product_obj = {
                "name": name,
                "category": category,
                "description": description,
                "price": price,
                # Include sku only if present
                **({"sku": sku} if sku else {}),
                "hsCode": hs_code,
                "countryOfOrigin": country,
                "weight": weight,
                "height": height,
                "length": length,
                "width": width,
                "quantity": quantity,
            }

            row_errs = _row_errors(idx + 2, product_obj)  # +2 because Excel data typically starts at row 2 (after header)
            if row_errs:
                errors.extend(row_errs)
            else:
                products.append(product_obj)

        if errors:
            friendly = _friendly_row_errors_message(errors, column_map)
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": errors,
                "friendly": friendly,
            })

        # Clean NaN/Inf in products to avoid JSON serialization errors
        def _clean(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, str) and obj.strip().lower() in ('nan', 'none'):
                return None
            if isinstance(obj, (list, tuple)):
                cleaned = [_clean(x) for x in obj if _clean(x) is not None]
                return cleaned if cleaned else None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if _clean(v) is not None}
            return obj

        safe_products = [_clean(p) for p in products]
        return {"products": safe_products}

    except HTTPException:
        raise
    except Exception as e:
        # Unexpected server error
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {str(e)}"})


@app.post("/parse-services")
async def parse_services(file: UploadFile = File(...)):
    """
    Parse and normalize service data from uploaded Excel or CSV file.
    Returns 200 with services[] ready for Node import, or 422 with row-level errors.
    """
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith((".xlsx", ".xls", ".csv")):
            return JSONResponse(status_code=400, content={"message": "Please upload an Excel file (.xlsx, .xls) or CSV file (.csv)."})

        content = await file.read()
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
        except Exception:
            return JSONResponse(status_code=400, content={"message": "We couldn't read that file. Please check the file format and try again."})

        # Drop fully empty rows/columns
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        if df.shape[0] == 0:
            friendly = {
                "title": "I couldn't find any data rows.",
                "bullets": ["Make sure your file has a header row and at least one data row."],
                "message": "I couldn't find any data rows. Make sure your file has a header row and at least one data row.",
            }
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": [{"row": None, "field": "file", "message": "No data rows found"}],
                "friendly": friendly,
            })

        # Build column map for services
        column_map = _build_service_column_map(df)

        # Validate required mappings exist
        missing_required_cols = [f for f in SERVICE_REQUIRED_FIELDS if column_map.get(f) is None]
        if missing_required_cols:
            friendly = _service_friendly_missing_columns_message(missing_required_cols)
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": [
                    {"row": None, "field": f, "message": "required column not found"}
                    for f in missing_required_cols
                ],
                "friendly": friendly,
            })

        # Build normalized services
        services: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx in range(len(df)):
            row_src = df.iloc[idx]

            def get_val(field: str) -> Any:
                col = column_map.get(field)
                if not col:
                    return None
                try:
                    val = row_src[col]
                    if pd.isna(val) or (isinstance(val, str) and val.strip().lower() in ('', 'nan', 'none')):
                        return None
                    return val
                except Exception:
                    return None

            # Extract and normalize service data
            service_name = str(get_val("serviceName")).strip() if get_val("serviceName") is not None else None
            category_raw = get_val("serviceCategory")
            service_category = _normalize_category(category_raw)
            description_raw = get_val("description")
            description = None if description_raw is None or str(description_raw).strip() == "" else str(description_raw).strip()
            price = _to_float(get_val("price"))

            service_obj = {
                "serviceName": service_name,
                "serviceCategory": service_category,
                "description": description,
                "price": price,
            }

            row_errs = _service_row_errors(idx + 2, service_obj)  # +2 because data typically starts at row 2 (after header)
            if row_errs:
                errors.extend(row_errs)
            else:
                services.append(service_obj)

        if errors:
            friendly = _service_friendly_row_errors_message(errors, column_map)
            return JSONResponse(status_code=422, content={
                "message": friendly["message"],
                "errors": errors,
                "friendly": friendly,
            })

        # Clean NaN/Inf in services to avoid JSON serialization errors
        def _clean(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, str) and obj.strip().lower() in ('nan', 'none'):
                return None
            if isinstance(obj, (list, tuple)):
                cleaned = [_clean(x) for x in obj if _clean(x) is not None]
                return cleaned if cleaned else None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if _clean(v) is not None}
            return obj

        safe_services = [_clean(s) for s in services]
        return {"services": safe_services}

    except HTTPException:
        raise
    except Exception as e:
        # Unexpected server error
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {str(e)}"})


@app.post("/parse/v2/services")
async def parse_v2_services_endpoint(file: UploadFile = File(...)):
    """V2 endpoint for parsing services with new schema"""
    return await parse_v2_services(file)


@app.post("/parse/v2/products")
async def parse_v2_products_endpoint(file: UploadFile = File(...)):
    """V2 endpoint for parsing products with new schema"""
    return await parse_v2_products(file)


# === Example Local Run (for dev testing) ===
if __name__ == "__main__":
    try:
        _load_model_if_needed(non_blocking=False)
        df = pd.read_excel("inventory_data.xlsx")
        # Simulate file processing via endpoint logic if needed
        print(f"Loaded {len(df)} rows for local testing")
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")