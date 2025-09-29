#!/usr/bin/env python3
"""
Startup script for the Deepfake Detection Python Backend
"""

import sys
import os
import subprocess

# Add python_backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_backend'))

try:
    from python_backend.app import app, socketio
    print("✓ Successfully imported Flask app")
    
    # Create necessary directories
    os.makedirs('python_backend/uploads', exist_ok=True)
    os.makedirs('python_backend/results', exist_ok=True)
    print("✓ Created upload and results directories")
    
    print("🚀 Starting Deepfake Detection Backend Server...")
    print("📡 Server running on http://0.0.0.0:8000")
    print("💻 WebSocket support enabled for real-time updates")
    print("⚡ Ready to process images and videos for deepfake detection")
    
    # Start the server on allowed port 8000 for Replit
    socketio.run(app, host='0.0.0.0', port=8000, debug=True, use_reloader=False, log_output=True)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing missing dependencies...")
    
    # Try to install missing packages
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors', 'opencv-python', 'mediapipe', 'scikit-learn', 'flask-socketio'], check=True)
    
    # Retry import
    from python_backend.app import app, socketio
    print("✓ Dependencies installed, starting server...")
    socketio.run(app, host='0.0.0.0', port=8000, debug=True, use_reloader=False, log_output=True)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)