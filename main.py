from fastapi import FastAPI, HTTPException, UploadFile, File
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import io
import os
import threading

# FastAPI app
app = FastAPI(title="Inventory Data Parser", description="AI-powered inventory data normalization service")

# Configure Hugging Face cache to ephemeral disk (safe for Heroku dynos)
os.environ.setdefault("HF_HOME", "/tmp/hf")

# === Target schema fields ===
standard_fields = [
    'product_name',
    'sku',
    'quantity',
    'price',
    'category',
    'description',
    'hs_code',
    'country_of_origin',
    'warehouse_location',
    'weight',
    'height',
    'length',
    'width',
    'image_url',
]

# === Synonym mapping for common header variations ===
field_synonyms = {
    "product_name": ["item", "items", "product", "product title", "product name", "name"],
    "sku": ["sku", "item code", "product code", "id", "identifier"],
    "quantity": ["qty", "quantity", "stock", "inventory"],
    "price": ["amount", "price", "cost", "value", "rate"],
    "category": ["group", "type", "department", "category", "section"],
    "description": ["details", "description", "info", "information", "specs"],
    "hs_code": ["hs code", "customs code", "tariff code", "import code", "export code"],
    "country_of_origin": ["country", "origin", "made in", "manufactured in", "produced in"],
    "warehouse_location": ["warehouse", "location", "storage", "stock", "inventory"],
    "weight": ["weight", "wt", "mass", "size", "dimension"],
    "height": ["height", "h", "size", "dimension"],
    "length": ["length", "l", "size", "dimension"],
    "width": ["width", "w", "size", "dimension"],
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


# === Semantic + synonym mapping function ===
def map_columns_semantic(df, standard_fields, threshold=0.5):
    user_columns = list(df.columns)
    column_map = {}

    # 1. Try matching based on field_synonyms
    for standard in standard_fields:
        synonyms = field_synonyms.get(standard, [])
        for user_col in user_columns:
            if user_col.lower() in [s.lower() for s in synonyms]:
                column_map[standard] = user_col
                break

    # 2. Prepare remaining fields and columns for embedding comparison
    unmatched_standards = [f for f in standard_fields if f not in column_map]
    unmatched_user_cols = [c for c in user_columns if c not in column_map.values()]

    if unmatched_standards and unmatched_user_cols:
        # Use column names + sample data to enrich embeddings
        user_col_samples = [
            f"{col}: {df[col].astype(str).dropna().head(3).tolist()}"
            for col in unmatched_user_cols
        ]

        # Ensure model is loaded (blocking here). If you prefer non-blocking, return 503 when not ready.
        if not _load_model_if_needed(non_blocking=False):
            raise HTTPException(status_code=503, detail="Model is warming up. Please retry shortly.")

        user_embeddings = _model.encode(user_col_samples, convert_to_tensor=True)
        schema_embeddings = _model.encode(unmatched_standards, convert_to_tensor=True)

        for i, schema_emb in enumerate(schema_embeddings):
            scores = util.cos_sim(schema_emb, user_embeddings)[0]
            best_score = scores.max().item()
            best_index = scores.argmax().item()

            if best_score >= threshold:
                column_map[unmatched_standards[i]] = unmatched_user_cols[best_index]

    return column_map


# === Normalize DataFrame based on matched columns ===
def normalize_inventory_data(df, column_map):
    normalized = {}
    for standard_field, user_column in column_map.items():
        if user_column in df.columns:
            normalized[standard_field] = df[user_column]
    return pd.DataFrame(normalized)


# === Helper: clean values for JSON ===
def clean_for_json(obj):
    if pd.isna(obj):
        return None
    elif isinstance(obj, (float, int)) and (pd.isna(obj) or (isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')))):
        return None
    return obj


@app.get("/")
async def root():
    return {"message": "Inventory Data Parser API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _model is not None, "model_loading": _model_loading}


@app.post("/parse-inventory")
async def parse_inventory(file: UploadFile = File(...)):
    """
    Parse and normalize inventory data from uploaded Excel file
    """
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")

        # Ensure model is ready. If still loading, tell client to retry.
        if not _load_model_if_needed(non_blocking=False):
            raise HTTPException(status_code=503, detail="Model is warming up. Please retry shortly.")

        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))

        user_columns = list(df.columns)
        column_map = map_columns_semantic(df, standard_fields)
        normalized_df = normalize_inventory_data(df, column_map)

        normalized_records = []
        for _, row in normalized_df.iterrows():
            clean_row = {col: clean_for_json(val) for col, val in row.items()}
            normalized_records.append(clean_row)

        return {
            "mapped_columns": column_map,
            "normalized_data": normalized_records,
            "original_columns": user_columns,
            "standard_fields": standard_fields
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# === Example Local Run (for dev testing) ===
if __name__ == "__main__":
    try:
        # Optional: eager load when running locally
        _load_model_if_needed(non_blocking=False)
        df = pd.read_excel("inventory_data.xlsx")
        column_map = map_columns_semantic(df, standard_fields)
        normalized_df = normalize_inventory_data(df, column_map)

        print("Mapped Columns:")
        print(column_map)
        print("\nNormalized DataFrame:")
        print(normalized_df)

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")