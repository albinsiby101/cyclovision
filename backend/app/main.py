"""
FastAPI Application Entrypoint for CycloVision
Initializes CORS middleware, routes, static asset serving, and life cycle events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.app.core.config import settings
from backend.app.api.routes.cyclone import router as cyclone_router

app = FastAPI(
    title="CycloVision API",
    description="AI-Powered Cyclone Pattern Intelligence & Early Warning System (SIH26070)",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins in development for seamless frontend binding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(cyclone_router)

# Serve static demo imagery
demo_dir = "data/demo/satellite_sequences"
if os.path.exists(demo_dir):
    app.mount("/static/demo", StaticFiles(directory=demo_dir), name="static_demo")

# Serve the production frontend build (single origin, same host as the API).
# `vite build` writes to frontend/dist; when present, it is served at "/" with
# index.html fallback so the SPA works at the root and deep links.
_app_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_app_dir, "..", ".."))  # …/cyclo/backend/app -> cyclo
_dist_dir = os.path.join(_repo_root, "frontend", "dist")
if os.path.isdir(_dist_dir):
    app.mount("/", StaticFiles(directory=_dist_dir, html=True), name="frontend")

@app.get("/")
def read_root():
    # When the production frontend build exists, serve the SPA at the root.
    _index = os.path.join(_dist_dir, "index.html") if os.path.isdir(_dist_dir) else None
    if _index and os.path.isfile(_index):
        from fastapi.responses import FileResponse
        return FileResponse(_index)
    return {
        "system": "CycloVision",
        "description": "AI-Powered Cyclone Pattern Intelligence & Early Warning System",
        "problem_statement": "SIH26070",
        "docs": "/docs",
        "health": "/health",
        "status": "/api/status"
    }