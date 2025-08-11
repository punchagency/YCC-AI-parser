from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import io
import os
import threading
import numpy as np
from typing import Dict, List, Any, Optional

# FastAPI app
app = FastAPI(title="Inventory Data Parser", description="AI-powered inventory data normalization service")

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

# === Synonyms for column name mapping ===
FIELD_SYNONYMS: Dict[str, List[str]] = {
    "name": ["name", "product name", "product", "item", "title"],
    "category": ["category", "categories", "group", "type", "department", "section"],
    "description": ["description", "details", "info", "information", "specs"],
    "price": ["price", "unit price", "amount", "cost", "value", "rate"],
    "sku": ["sku", "item code", "product code", "code", "id", "identifier"],
    "hsCode": ["hs code", "hscode", "hs", "tariff code", "customs code"],
    "countryOfOrigin": ["country of origin", "origin", "country", "made in", "manufactured in"],
    "weight": ["weight", "wt", "mass"],
    "height": ["height", "h"],
    "length": ["length", "l"],
    "width": ["width", "w"],
    "quantity": ["quantity", "qty", "stock", "inventory", "available"],
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
            return float(value)
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if s == "":
                return None
            # Extract first numeric token (optional sign, digits, optional decimal)
            import re
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            if not m:
                return None
            return float(m.group(0))
        # Fallback: try casting generically
        return float(value)
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
def _build_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    user_columns = list(df.columns)
    lower_user = {c.lower(): c for c in user_columns}

    column_map: Dict[str, Optional[str]] = {field: None for field in TARGET_FIELDS}

    # 1) Exact/synonym match first
    for field, synonyms in FIELD_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in lower_user:
                column_map[field] = lower_user[syn.lower()]
                break

    # 2) Semantic fallback for unmapped fields
    unmapped_fields = [f for f, c in column_map.items() if c is None]
    candidate_columns = [c for c in user_columns if c not in column_map.values()]

    if unmapped_fields and candidate_columns:
        if not _load_model_if_needed(non_blocking=False):
            # Model should be loaded by now; if not, treat as missing mappings
            return column_map

        # Describe candidate columns with small data samples for better signals
        user_col_samples = []
        for col in candidate_columns:
            try:
                sample = df[col].astype(str).dropna().head(3).tolist()
            except Exception:
                sample = []
            user_col_samples.append(f"{col}: {sample}")

        user_emb = _model.encode(user_col_samples, convert_to_numpy=True)
        field_emb = _model.encode(unmapped_fields, convert_to_numpy=True)

        # Normalize
        user_norm = np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-12
        field_norm = np.linalg.norm(field_emb, axis=1, keepdims=True) + 1e-12
        user_n = user_emb / user_norm
        field_n = field_emb / field_norm

        threshold = 0.5  # conservative
        for i, field in enumerate(unmapped_fields):
            scores = (user_n @ field_n[i].reshape(-1, 1)).ravel()
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                column_map[field] = candidate_columns[best_idx]

        # Free large arrays
        del user_emb, field_emb, user_n, field_n

    return column_map


def _normalize_category(value: Any) -> List[str]:
    if value is None:
        return ["Other"]
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["Other"]
    text = str(value).strip()
    if not text:
        return ["Other"]
    # Split on common delimiters
    parts = [p.strip() for p in re_split(text)]
    parts = [p for p in parts if p]
    return parts or ["Other"]


def re_split(text: str) -> List[str]:
    # Helper separated to avoid importing re at module import (keeps startup fast)
    import re
    return re.split(r"[,;|]", text)


def _row_errors(row_idx: int, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []

    def add(field: str, msg: str):
        errs.append({"row": row_idx, "field": field, "message": msg})

    # Required presence
    for field in REQUIRED_FIELDS:
        if row.get(field) in (None, ""):
            add(field, "required")

    # Type/constraints
    if row.get("price") is not None and (not isinstance(row["price"], (int, float)) or row["price"] < 0):
        add("price", "must be a non-negative number")
    for dim in ["weight", "height", "length", "width"]:
        if row.get(dim) is not None and (not isinstance(row[dim], (int, float)) or row[dim] < 0):
            add(dim, "must be a non-negative number")
    if row.get("quantity") is not None and (not isinstance(row["quantity"], int) or row["quantity"] < 0):
        add("quantity", "must be a non-negative integer")

    # Category must be non-empty array
    if not isinstance(row.get("category"), list) or len(row["category"]) == 0:
        add("category", "must be a non-empty list")

    return errs


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
            return JSONResponse(status_code=400, content={"message": "File must be an Excel file (.xlsx or .xls)"})

        content = await file.read()
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception:
            return JSONResponse(status_code=400, content={"message": "Failed to read Excel file"})

        # Drop fully empty rows/columns
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        if df.shape[0] == 0:
            return JSONResponse(status_code=422, content={"errors": [{"row": None, "field": "file", "message": "No data rows found"}]})

        # Build column map
        column_map = _build_column_map(df)

        # Validate required mappings exist
        missing_required_cols = [f for f in REQUIRED_FIELDS if column_map.get(f) is None]
        if missing_required_cols:
            return JSONResponse(status_code=422, content={
                "errors": [
                    {"row": None, "field": f, "message": "required column not found"}
                    for f in missing_required_cols
                ]
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
                    return row_src[col]
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
            return JSONResponse(status_code=422, content={"errors": errors})

        return {"products": products}

    except HTTPException:
        raise
    except Exception as e:
        # Unexpected server error
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {str(e)}"})


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