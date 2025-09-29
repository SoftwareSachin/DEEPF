#!/usr/bin/env python3
"""
Simplified Python backend for testing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import uuid

app = Flask(__name__)
CORS(app, origins=["*"])

# Store for demo results
results_store = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Simulate processing with mock results
    file_type = 'image' if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')) else 'video'
    
    # Store mock result
    results_store[job_id] = {
        'overall_prediction': 'REAL',
        'confidence': 0.85,
        'fileName': file.filename,
        'fileType': file_type,
        'faces_detected': 1,
        'face_results': [{
            'face_id': 1,
            'bbox': [50, 50, 150, 150],
            'prediction': 'REAL',
            'confidence': 0.85
        }],
        'analysis_timestamp': time.time()
    }
    
    return jsonify({
        'job_id': job_id,
        'filename': file.filename,
        'file_type': file_type,
        'status': 'uploaded'
    })

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    if job_id not in results_store:
        return jsonify({'error': 'Job not found'}), 404
    
    result = results_store[job_id]
    return jsonify({
        'job_id': job_id,
        'filename': result['fileName'],
        'file_type': result['fileType'],
        'results': result
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Deepfake Detection Backend...")
    print("📡 Server running on http://0.0.0.0:5001")
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()