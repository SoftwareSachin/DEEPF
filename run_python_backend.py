#!/usr/bin/env python3
"""
Startup script for the Python backend server.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the python_backend directory to the Python path
backend_dir = Path(__file__).parent / "python_backend"
sys.path.insert(0, str(backend_dir))

try:
    from app import app
except ImportError:
    # Fallback for LSP and development
    from python_backend.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    
    print(f"Starting Deepfake Detection API on port {port}")
    print("Available endpoints:")
    print("  POST /api/upload - Upload file for analysis")
    print("  POST /api/analyze/{file_id} - Analyze uploaded file")
    print("  WS /ws - WebSocket for real-time updates")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # Disable in production
    )