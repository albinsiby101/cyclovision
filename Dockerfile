# ---- Stage 1: build the React frontend ----
FROM node:20-alpine AS fe
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime ----
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY models models
COPY data data
COPY ml ml
COPY .env.example .env.example

# Production build of the SPA (served by FastAPI from the same origin)
COPY --from=fe /fe/dist frontend/dist

EXPOSE 8000
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}