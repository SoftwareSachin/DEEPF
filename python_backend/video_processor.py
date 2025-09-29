"""
Video Processor Module
Handles video frame extraction and processing for deepfake detection
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Callable, Optional
from datetime import datetime
import json

class VideoProcessor:
    def __init__(self):
        """Initialize video processor"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        self.max_frames = 30  # Limit frames to process for efficiency
        self.frame_skip = 5   # Process every 5th frame
        
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get basic information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0,
            'codec': 'unknown'
        }
        
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        
        cap.release()
        return info
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None, 
                      frame_skip: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video for processing
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            frame_skip: Skip every N frames
            
        Returns:
            List of frame images as numpy arrays
        """
        if max_frames is None:
            max_frames = self.max_frames
        if frame_skip is None:
            frame_skip = self.frame_skip
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames according to frame_skip parameter
            if frame_count % frame_skip == 0:
                frames.append(frame.copy())
                extracted_count += 1
                
                # Stop if we've extracted enough frames
                if extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Process a complete video for deepfake detection
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of frame analysis results
        """
        from face_extractor import FaceExtractor
        from deepfake_detector import DeepfakeDetector
        
        # Initialize processors
        face_extractor = FaceExtractor()
        deepfake_detector = DeepfakeDetector()
        
        # Get video info
        try:
            video_info = self.get_video_info(video_path)
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error reading video: {str(e)}")
            return []
        
        if progress_callback:
            progress_callback(10, f"Video loaded: {video_info['total_frames']} frames")
        
        # Extract frames
        try:
            frames = self.extract_frames(video_path)
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error extracting frames: {str(e)}")
            return []
        
        if not frames:
            if progress_callback:
                progress_callback(0, "No frames extracted from video")
            return []
        
        if progress_callback:
            progress_callback(30, f"Extracted {len(frames)} frames for analysis")
        
        # Process each frame
        frame_results = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            try:
                # Update progress
                frame_progress = 30 + (i / total_frames) * 50
                if progress_callback:
                    progress_callback(frame_progress, f"Processing frame {i+1}/{total_frames}")
                
                # Extract faces from frame
                faces = face_extractor.extract_faces(frame)
                
                if not faces:
                    # No faces in this frame
                    frame_results.append({
                        'frame_number': i,
                        'timestamp': i * self.frame_skip / video_info['fps'] if video_info['fps'] > 0 else 0,
                        'faces_detected': 0,
                        'face_results': [],
                        'status': 'no_faces'
                    })
                    continue
                
                # Analyze each face in the frame
                face_results = []
                for j, face_data in enumerate(faces):
                    face_image = face_data['face']
                    bbox = face_data['bbox']
                    
                    # Run deepfake detection
                    prediction, confidence = deepfake_detector.predict(face_image)
                    
                    face_results.append({
                        'face_id': j + 1,
                        'bbox': bbox,
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'detection_confidence': float(face_data['confidence'])
                    })
                
                # Store frame result
                frame_results.append({
                    'frame_number': i,
                    'timestamp': i * self.frame_skip / video_info['fps'] if video_info['fps'] > 0 else 0,
                    'faces_detected': len(faces),
                    'face_results': face_results,
                    'status': 'processed'
                })
                
            except Exception as e:
                # Error processing this frame
                frame_results.append({
                    'frame_number': i,
                    'timestamp': i * self.frame_skip / video_info['fps'] if video_info['fps'] > 0 else 0,
                    'faces_detected': 0,
                    'face_results': [],
                    'status': 'error',
                    'error': str(e)
                })
                
                if progress_callback:
                    progress_callback(frame_progress, f"Error in frame {i+1}: {str(e)}")
        
        if progress_callback:
            progress_callback(80, "Frame analysis complete")
        
        return frame_results
    
    def create_annotated_video(self, video_path: str, frame_results: List[Dict], 
                              output_path: str) -> bool:
        """
        Create an annotated video with detection results overlaid
        
        Args:
            video_path: Original video path
            frame_results: Results from process_video
            output_path: Path for output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            result_index = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if this frame has results
                if (result_index < len(frame_results) and 
                    frame_results[result_index]['frame_number'] * self.frame_skip <= frame_count):
                    
                    result = frame_results[result_index]
                    
                    # Draw face detections and predictions
                    for face_result in result['face_results']:
                        bbox = face_result['bbox']
                        prediction = face_result['prediction']
                        confidence = face_result['confidence']
                        
                        # Choose color based on prediction
                        color = (0, 0, 255) if prediction == 'FAKE' else (0, 255, 0)  # Red for fake, green for real
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        
                        # Draw prediction label
                        label = f"{prediction}: {confidence:.2f}"
                        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    result_index += 1
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            print(f"Error creating annotated video: {e}")
            return False
    
    def get_frame_thumbnails(self, frames: List[np.ndarray], thumbnail_size: tuple = (150, 150)) -> List[np.ndarray]:
        """
        Create thumbnail versions of frames for display
        
        Args:
            frames: List of frame images
            thumbnail_size: Target thumbnail size
            
        Returns:
            List of thumbnail images
        """
        thumbnails = []
        
        for frame in frames:
            # Resize frame to thumbnail size
            thumbnail = cv2.resize(frame, thumbnail_size)
            thumbnails.append(thumbnail)
        
        return thumbnails
    
    def save_frame_results(self, frame_results: List[Dict], output_path: str):
        """
        Save frame analysis results to JSON file
        
        Args:
            frame_results: Results from process_video
            output_path: Path to save JSON file
        """
        try:
            # Prepare data for JSON serialization
            json_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_frames_analyzed': len(frame_results),
                'frame_results': frame_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving frame results: {e}")
    
    def calculate_video_stats(self, frame_results: List[Dict]) -> Dict:
        """
        Calculate statistics from video analysis results
        
        Args:
            frame_results: Results from process_video
            
        Returns:
            Dictionary with video analysis statistics
        """
        if not frame_results:
            return {
                'total_frames': 0,
                'frames_with_faces': 0,
                'total_faces': 0,
                'fake_predictions': 0,
                'real_predictions': 0,
                'average_confidence': 0.0,
                'fake_percentage': 0.0
            }
        
        total_frames = len(frame_results)
        frames_with_faces = sum(1 for result in frame_results if result['faces_detected'] > 0)
        
        all_predictions = []
        all_confidences = []
        
        for result in frame_results:
            for face_result in result['face_results']:
                all_predictions.append(face_result['prediction'])
                all_confidences.append(face_result['confidence'])
        
        fake_count = sum(1 for pred in all_predictions if pred == 'FAKE')
        real_count = len(all_predictions) - fake_count
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        fake_percentage = (fake_count / len(all_predictions) * 100) if all_predictions else 0.0
        
        return {
            'total_frames': total_frames,
            'frames_with_faces': frames_with_faces,
            'total_faces': len(all_predictions),
            'fake_predictions': fake_count,
            'real_predictions': real_count,
            'average_confidence': float(avg_confidence),
            'fake_percentage': float(fake_percentage)
        }