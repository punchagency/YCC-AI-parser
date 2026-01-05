from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import pandas as pd
import io
import numpy as np
from typing import Dict, List, Any, Optional
import math
import difflib
import re
from collections import defaultdict

# V2 Target fields
V2_SERVICE_FIELDS = ["name", "description", "price", "categoryName", "isQuotable"]
V2_PRODUCT_FIELDS = ["name", "description", "price", "categoryName", "sku", "quantity", "minRestockLevel", 
                     "street", "city", "zipcode", "state", "country", "hsCode", "weight", "length", "width", "height"]

# Required fields
V2_SERVICE_REQUIRED = ["name", "price", "categoryName"]
V2_PRODUCT_REQUIRED = ["name", "price", "categoryName"]

# Synonyms
V2_SERVICE_SYNONYMS = {
    "name": ["name", "service name", "service", "title", "service title"],
    "description": ["description", "details", "info", "service description"],
    "price": ["price", "cost", "rate", "fee", "amount"],
    "categoryName": ["category name", "categoryname", "category", "service category", "type"],
    "isQuotable": ["is quotable", "isquotable", "quotable", "quote", "requires quote"],
}

V2_PRODUCT_SYNONYMS = {
    "name": ["name", "product name", "product", "item", "title"],
    "description": ["description", "details", "info", "product description"],
    "price": ["price", "cost", "amount", "unit price"],
    "categoryName": ["category name", "categoryname", "category", "product category", "type"],
    "sku": ["sku", "item code", "product code", "code"],
    "quantity": ["quantity", "qty", "stock", "inventory"],
    "minRestockLevel": ["min restock level", "minrestocklevel", "restock level", "minimum stock", "min stock"],
    "street": ["street", "address", "street address", "address line"],
    "city": ["city", "town"],
    "zipcode": ["zipcode", "zip code", "zip", "postal code"],
    "state": ["state", "province", "region"],
    "country": ["country", "nation"],
    "hsCode": ["hs code", "hscode", "hs", "tariff code"],
    "weight": ["weight", "wt", "mass"],
    "length": ["length", "l"],
    "width": ["width", "w"],
    "height": ["height", "h"],
}

_model_cache = None

def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _model_cache

def _normalize_header(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\s_\-]+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return re.sub(r"\s+", " ", t).strip()

def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            f = float(value)
            return f if math.isfinite(f) else None
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if s == "":
                return None
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            if not m:
                return None
            f = float(m.group(0))
            return f if math.isfinite(f) else None
        return float(value) if math.isfinite(float(value)) else None
    except:
        return None

def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    return int(round(f)) if f is not None else None

def _to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y")
    return bool(value)

def _build_column_map_v2(df: pd.DataFrame, target_fields: List[str], synonyms: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    user_columns = list(df.columns)
    normalized_to_original = {_normalize_header(c): c for c in user_columns}
    column_map = {field: None for field in target_fields}
    
    # Exact match
    normalized_synonyms = {f: [_normalize_header(s) for s in syns] for f, syns in synonyms.items()}
    for field, syns in normalized_synonyms.items():
        for h_norm in normalized_to_original.keys():
            if h_norm in syns:
                column_map[field] = normalized_to_original[h_norm]
                break
    
    # Fuzzy + semantic match
    unmapped = [f for f, c in column_map.items() if c is None]
    candidates = [c for c in user_columns if c not in column_map.values()]
    
    if unmapped and candidates:
        model = _get_model()
        scores = defaultdict(float)
        
        for field in unmapped:
            for col in candidates:
                # String similarity
                best_str = max([difflib.SequenceMatcher(None, _normalize_header(col), _normalize_header(s)).ratio() 
                               for s in synonyms.get(field, [field])] or [0.0])
                scores[(field, col)] += 0.4 * best_str
        
        # Semantic similarity
        cand_emb = model.encode([_normalize_header(c) for c in candidates], convert_to_numpy=True)
        cand_norm = np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-12
        cand_n = cand_emb / cand_norm
        
        for field in unmapped:
            phrases = synonyms.get(field, [field])
            field_emb = model.encode([_normalize_header(p) for p in phrases], convert_to_numpy=True)
            field_norm = np.linalg.norm(field_emb, axis=1, keepdims=True) + 1e-12
            field_n = field_emb / field_norm
            cos = cand_n @ field_n.T
            best_per_cand = cos.max(axis=1)
            for idx, col in enumerate(candidates):
                scores[(field, col)] += 0.6 * float(best_per_cand[idx])
        
        # Assign greedily
        used_fields, used_cols = set(), set()
        for (f, c), s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if s < 0.7 or f in used_fields or c in used_cols:
                continue
            column_map[f] = c
            used_fields.add(f)
            used_cols.add(c)
    
    return column_map

def _validate_v2_service(row_idx: int, row: Dict[str, Any], raw_value: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs = []
    for field in V2_SERVICE_REQUIRED:
        if row.get(field) is None:
            errs.append({
                "row": row_idx,
                "field": field,
                "value": raw_value.get(field),
                "message": f"Required field '{field}' is missing or empty at row {row_idx}"
            })
    if row.get("price") is not None and (not isinstance(row["price"], (int, float)) or row["price"] < 0):
        errs.append({
            "row": row_idx,
            "field": "price",
            "value": raw_value.get("price"),
            "message": f"Field 'price' at row {row_idx} must be a non-negative number, got: {raw_value.get('price')}"
        })
    return errs

def _validate_v2_product(row_idx: int, row: Dict[str, Any], raw_value: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs = []
    for field in V2_PRODUCT_REQUIRED:
        if row.get(field) is None:
            errs.append({
                "row": row_idx,
                "field": field,
                "value": raw_value.get(field),
                "message": f"Required field '{field}' is missing or empty at row {row_idx}"
            })
    if row.get("price") is not None and (not isinstance(row["price"], (int, float)) or row["price"] < 0):
        errs.append({
            "row": row_idx,
            "field": "price",
            "value": raw_value.get("price"),
            "message": f"Field 'price' at row {row_idx} must be a non-negative number, got: {raw_value.get('price')}"
        })
    for dim in ["weight", "length", "width", "height"]:
        if row.get(dim) is not None and (not isinstance(row[dim], (int, float)) or row[dim] < 0):
            errs.append({
                "row": row_idx,
                "field": dim,
                "value": raw_value.get(dim),
                "message": f"Field '{dim}' at row {row_idx} must be a non-negative number, got: {raw_value.get(dim)}"
            })
    if row.get("quantity") is not None and (not isinstance(row["quantity"], int) or row["quantity"] < 0):
        errs.append({
            "row": row_idx,
            "field": "quantity",
            "value": raw_value.get("quantity"),
            "message": f"Field 'quantity' at row {row_idx} must be a non-negative integer, got: {raw_value.get('quantity')}"
        })
    return errs

async def parse_v2_services(file: UploadFile = File(...)):
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith((".xlsx", ".xls", ".csv")):
            return JSONResponse(status_code=400, content={"message": "Upload Excel or CSV file"})
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(content))
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if df.shape[0] == 0:
            return JSONResponse(status_code=422, content={"message": "No data rows found"})
        
        column_map = _build_column_map_v2(df, V2_SERVICE_FIELDS, V2_SERVICE_SYNONYMS)
        missing = [f for f in V2_SERVICE_REQUIRED if column_map.get(f) is None]
        if missing:
            return JSONResponse(status_code=422, content={"message": f"Missing required columns: {', '.join(missing)}"})
        
        services, errors = [], []
        for idx in range(len(df)):
            row_src = df.iloc[idx]
            def get_val(field: str):
                col = column_map.get(field)
                if not col:
                    return None
                val = row_src[col]
                return None if pd.isna(val) or (isinstance(val, str) and val.strip() == "") else val
            
            raw_values = {field: get_val(field) for field in V2_SERVICE_FIELDS}
            
            service = {
                "name": str(get_val("name")).strip() if get_val("name") else None,
                "description": str(get_val("description")).strip() if get_val("description") else None,
                "price": _to_float(get_val("price")),
                "categoryName": str(get_val("categoryName")).strip() if get_val("categoryName") else None,
                "isQuotable": _to_bool(get_val("isQuotable")),
            }
            
            row_errs = _validate_v2_service(idx + 2, service, raw_values)
            if row_errs:
                errors.extend(row_errs)
            else:
                services.append(service)
        
        if errors:
            return JSONResponse(status_code=422, content={
                "message": f"Validation failed: {len(errors)} error(s) found in the uploaded file",
                "errors": errors,
                "totalErrors": len(errors)
            })
        
        return {"services": services}
    
    except pd.errors.EmptyDataError:
        return JSONResponse(status_code=422, content={"message": "The uploaded file is empty or contains no valid data"})
    except pd.errors.ParserError as e:
        return JSONResponse(status_code=422, content={"message": f"Failed to parse file: {str(e)}. Please ensure the file format is correct"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Unexpected error processing services file: {str(e)}"})

async def parse_v2_products(file: UploadFile = File(...)):
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith((".xlsx", ".xls", ".csv")):
            return JSONResponse(status_code=400, content={"message": "Upload Excel or CSV file"})
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(content))
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if df.shape[0] == 0:
            return JSONResponse(status_code=422, content={"message": "No data rows found"})
        
        column_map = _build_column_map_v2(df, V2_PRODUCT_FIELDS, V2_PRODUCT_SYNONYMS)
        missing = [f for f in V2_PRODUCT_REQUIRED if column_map.get(f) is None]
        if missing:
            return JSONResponse(status_code=422, content={"message": f"Missing required columns: {', '.join(missing)}"})
        
        products, errors = [], []
        for idx in range(len(df)):
            row_src = df.iloc[idx]
            def get_val(field: str):
                col = column_map.get(field)
                if not col:
                    return None
                val = row_src[col]
                return None if pd.isna(val) or (isinstance(val, str) and val.strip() == "") else val
            
            raw_values = {field: get_val(field) for field in V2_PRODUCT_FIELDS}
            
            product = {
                "name": str(get_val("name")).strip() if get_val("name") else None,
                "description": str(get_val("description")).strip() if get_val("description") else None,
                "price": _to_float(get_val("price")),
                "categoryName": str(get_val("categoryName")).strip() if get_val("categoryName") else None,
                "sku": str(get_val("sku")).strip() if get_val("sku") else None,
                "quantity": _to_int(get_val("quantity")),
                "minRestockLevel": _to_int(get_val("minRestockLevel")),
                "street": str(get_val("street")).strip() if get_val("street") else None,
                "city": str(get_val("city")).strip() if get_val("city") else None,
                "zipcode": str(get_val("zipcode")).strip() if get_val("zipcode") else None,
                "state": str(get_val("state")).strip() if get_val("state") else None,
                "country": str(get_val("country")).strip() if get_val("country") else None,
                "hsCode": str(get_val("hsCode")).strip() if get_val("hsCode") else None,
                "weight": _to_float(get_val("weight")),
                "length": _to_float(get_val("length")),
                "width": _to_float(get_val("width")),
                "height": _to_float(get_val("height")),
            }
            
            row_errs = _validate_v2_product(idx + 2, product, raw_values)
            if row_errs:
                errors.extend(row_errs)
            else:
                products.append(product)
        
        if errors:
            return JSONResponse(status_code=422, content={
                "message": f"Validation failed: {len(errors)} error(s) found in the uploaded file",
                "errors": errors,
                "totalErrors": len(errors)
            })
        
        return {"products": products}
    
    except pd.errors.EmptyDataError:
        return JSONResponse(status_code=422, content={"message": "The uploaded file is empty or contains no valid data"})
    except pd.errors.ParserError as e:
        return JSONResponse(status_code=422, content={"message": f"Failed to parse file: {str(e)}. Please ensure the file format is correct"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Unexpected error processing products file: {str(e)}"})
