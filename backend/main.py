"""Entry point kept for existing deploy commands:

    uvicorn backend.main:app

The application itself lives in backend/app/.
"""

from backend.app.main import app  # noqa: F401
