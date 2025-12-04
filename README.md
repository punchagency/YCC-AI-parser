# YCC AI Parser

AI-powered inventory and service data normalization API built with FastAPI.

## Prerequisites

- Python 3.11+ (see `runtime.txt`)
- pip (Python package manager)

## Development Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

```bash
source venv/bin/activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

**If you encounter an AssertionError during installation:**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 5. Run the development server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### V1 Endpoints
- `GET /` - API status
- `GET /health` - Health check with model status
- `POST /parse-inventory` - Parse Excel inventory files (.xlsx, .xls)
- `POST /parse-services` - Parse Excel/CSV service files (.xlsx, .xls, .csv)

### V2 Endpoints
- `POST /parse/v2/services` - Parse services with fields: name, description, price, categoryName, isQuotable
- `POST /parse/v2/products` - Parse products with fields: name, description, price, categoryName, sku, quantity, minRestockLevel, street, city, zipcode, state, country, hsCode, weight, length, width, height

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# V1 endpoints
curl -X POST -F "file=@inventory_data.xlsx" http://localhost:8000/parse-inventory
curl -X POST -F "file=@test_services.csv" http://localhost:8000/parse-services

# V2 endpoints
curl -X POST -F "file=@services.xlsx" http://localhost:8000/parse/v2/services
curl -X POST -F "file=@products.csv" http://localhost:8000/parse/v2/products
```

### Using the interactive docs

Visit `http://localhost:8000/docs` for Swagger UI

## Environment Variables

- `CORS_ORIGINS` - Comma-separated list of allowed origins (default: `http://localhost:3000,https://yachtcrewcenter.com`)
- `HF_HOME` - Hugging Face cache directory (default: `/tmp/hf`)

## Notes

- First request may be slow as the AI model loads
- Uses CPU-only PyTorch for compatibility
- Supports flexible column mapping with AI-powered field detection
