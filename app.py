"""
Hugging Face Spaces entrypoint.

Spaces with `sdk: gradio` run `python app.py` and reverse-proxy whatever HTTP
server listens on $PORT (default 7860). CycloVision does not use a Gradio UI -
this simply boots the real FastAPI application (SPA + /api on the same origin)
inside the free CPU Space runtime.
"""
import os

import uvicorn

from backend.app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))