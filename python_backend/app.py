#!/usr/bin/env python3
"""
Deepfake Detection Backend Server
Lightweight implementation using OpenCV and MediaPipe for face detection
with ensemble CNN models for deepfake detection without TensorFlow
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import json
import threading
import time
from datetime import datetime
import uuid

from .video_processor import VideoProcessor
from .deepfake_detector import DeepfakeDetector
from .face_extractor import FaceExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize processors
video_processor = VideoProcessor()
deepfake_detector = DeepfakeDetector()
face_extractor = FaceExtractor()

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store processing jobs
processing_jobs = {}

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename or 'unknown')
        
        # Save file with job ID prefix
        safe_filename = f"{job_id}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        
        # Determine file type
        file_extension = filename.rsplit('.', 1)[1].lower()
        file_type = 'image' if file_extension in ['jpg', 'jpeg', 'png'] else 'video'
        
        # Create job entry
        processing_jobs[job_id] = {
            'id': job_id,
            'filename': filename,
            'file_path': file_path,
            'file_type': file_type,
            'status': 'uploaded',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'results': None,
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_file, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'file_type': file_type,
            'status': 'uploaded'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get processing status for a job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'file_type': job['file_type'],
        'filename': job['filename'],
        'created_at': job['created_at'],
        'results': job['results'],
        'error': job['error']
    })

@app.route('/api/results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """Get detailed results for a completed job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    return jsonify({
        'job_id': job_id,
        'filename': job['filename'],
        'file_type': job['file_type'],
        'results': job['results'],
        'created_at': job['created_at']
    })

@app.route('/api/download/<path:filename>')
def download_file(filename):
    """Serve processed result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

def process_file(job_id):
    """Background processing function"""
    try:
        job = processing_jobs[job_id]
        job['status'] = 'processing'
        emit_progress(job_id, 10, 'Starting analysis...')
        
        file_path = job['file_path']
        file_type = job['file_type']
        
        if file_type == 'image':
            results = process_image(job_id, file_path)
        else:
            results = process_video(job_id, file_path)
        
        job['status'] = 'completed'
        job['progress'] = 100
        job['results'] = results
        emit_progress(job_id, 100, 'Analysis complete!')
        
    except Exception as e:
        job = processing_jobs[job_id]
        job['status'] = 'failed'
        job['error'] = str(e)
        emit_progress(job_id, 0, f'Error: {str(e)}')
        print(f"Processing error for job {job_id}: {e}")

def process_image(job_id, file_path):
    """Process a single image"""
    emit_progress(job_id, 20, 'Loading image...')
    
    # Load image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Could not load image")
    
    emit_progress(job_id, 40, 'Detecting faces...')
    
    # Extract faces
    faces = face_extractor.extract_faces(image)
    
    if not faces:
        return {
            'overall_prediction': 'UNKNOWN',
            'confidence': 0.0,
            'faces_detected': 0,
            'face_results': [],
            'message': 'No faces detected in the image'
        }
    
    emit_progress(job_id, 70, f'Analyzing {len(faces)} faces...')
    
    # Analyze each face
    face_results = []
    predictions = []
    confidences = []
    
    for i, face_data in enumerate(faces):
        face_image = face_data['face']
        bbox = face_data['bbox']
        
        # Run deepfake detection
        prediction, confidence = deepfake_detector.predict(face_image)
        
        face_results.append({
            'face_id': i + 1,
            'bbox': bbox,
            'prediction': prediction,
            'confidence': float(confidence),
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        })
        
        predictions.append(prediction)
        confidences.append(confidence)
    
    # Calculate overall result
    fake_count = sum(1 for p in predictions if p == 'FAKE')
    overall_prediction = 'FAKE' if fake_count > len(predictions) / 2 else 'REAL'
    overall_confidence = np.mean(confidences)
    
    emit_progress(job_id, 90, 'Generating results...')
    
    return {
        'overall_prediction': overall_prediction,
        'confidence': float(overall_confidence),
        'faces_detected': len(faces),
        'face_results': face_results,
        'analysis_timestamp': datetime.now().isoformat()
    }

def process_video(job_id, file_path):
    """Process a video file"""
    emit_progress(job_id, 20, 'Loading video...')
    
    # Extract frames and analyze
    frame_results = video_processor.process_video(
        file_path, 
        progress_callback=lambda p, msg: emit_progress(job_id, 20 + p * 0.6, msg)
    )
    
    if not frame_results:
        return {
            'overall_prediction': 'UNKNOWN',
            'confidence': 0.0,
            'total_frames': 0,
            'frames_with_faces': 0,
            'frame_results': [],
            'message': 'No faces detected in any frames'
        }
    
    emit_progress(job_id, 80, 'Analyzing frame results...')
    
    # Aggregate results across all frames
    all_predictions = []
    all_confidences = []
    frames_with_faces = 0
    
    for frame_result in frame_results:
        if frame_result['faces_detected'] > 0:
            frames_with_faces += 1
            # Take the highest confidence prediction from each frame
            best_face = max(frame_result['face_results'], key=lambda x: x['confidence'])
            all_predictions.append(best_face['prediction'])
            all_confidences.append(best_face['confidence'])
    
    if not all_predictions:
        overall_prediction = 'UNKNOWN'
        overall_confidence = 0.0
    else:
        fake_count = sum(1 for p in all_predictions if p == 'FAKE')
        overall_prediction = 'FAKE' if fake_count > len(all_predictions) / 2 else 'REAL'
        overall_confidence = np.mean(all_confidences)
    
    emit_progress(job_id, 95, 'Finalizing results...')
    
    return {
        'overall_prediction': overall_prediction,
        'confidence': float(overall_confidence),
        'total_frames': len(frame_results),
        'frames_with_faces': frames_with_faces,
        'frame_results': frame_results[:10],  # Limit to first 10 frames for display
        'analysis_timestamp': datetime.now().isoformat()
    }

def emit_progress(job_id, progress, message):
    """Emit progress update via WebSocket"""
    socketio.emit('progress', {
        'job_id': job_id,
        'progress': progress,
        'message': message
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting Deepfake Detection Server...")
    print("Server will run on http://0.0.0.0:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False, log_output=True)