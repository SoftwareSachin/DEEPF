"""
Face Extractor Module
Uses MediaPipe for lightweight face detection and OpenCV for face extraction
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple

class FaceExtractor:
    def __init__(self):
        """Initialize MediaPipe face detection"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range, 0 for short range
            min_detection_confidence=0.5
        )
    
    def extract_faces(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> List[Dict]:
        """
        Extract faces from an image using MediaPipe
        
        Args:
            image: Input image as numpy array
            target_size: Size to resize extracted faces to
        
        Returns:
            List of dictionaries containing face data and bounding boxes
        """
        if image is None:
            return []
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounding box is within image bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + width)
                y2 = min(h, y + height)
                
                # Extract face region
                face_region = image[y:y2, x:x2]
                
                if face_region.size > 0:
                    # Resize face to target size
                    face_resized = cv2.resize(face_region, target_size)
                    
                    faces.append({
                        'face': face_resized,
                        'bbox': [x, y, x2, y2],
                        'confidence': detection.score[0],
                        'original_face': face_region
                    })
        
        return faces
    
    def extract_faces_batch(self, images: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> List[List[Dict]]:
        """
        Extract faces from multiple images
        
        Args:
            images: List of input images
            target_size: Size to resize extracted faces to
        
        Returns:
            List of face extraction results for each image
        """
        results = []
        for image in images:
            faces = self.extract_faces(image, target_size)
            results.append(faces)
        return results
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict], show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: Input image
            faces: List of face detection results
            show_confidence: Whether to show confidence scores
        
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for i, face_data in enumerate(faces):
            bbox = face_data['bbox']
            confidence = face_data.get('confidence', 0.0)
            
            # Draw bounding box
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw face ID and confidence
            if show_confidence:
                label = f"Face {i+1}: {confidence:.2f}"
                cv2.putText(result_image, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_image
    
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for deepfake detection
        
        Args:
            face: Face image as numpy array
        
        Returns:
            Preprocessed face image
        """
        # Normalize pixel values to [0, 1]
        face_normalized = face.astype(np.float32) / 255.0
        
        # Apply histogram equalization to improve contrast
        if len(face.shape) == 3:
            # Convert to grayscale for equalization
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            # Convert back to 3 channels
            equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            # Blend with original
            face_enhanced = cv2.addWeighted(face_normalized, 0.7, equalized_bgr.astype(np.float32) / 255.0, 0.3, 0)
        else:
            face_enhanced = face_normalized
        
        return face_enhanced
    
    def get_face_landmarks(self, image: np.ndarray) -> List[Dict]:
        """
        Extract facial landmarks using MediaPipe Face Mesh
        
        Args:
            image: Input image
        
        Returns:
            List of landmark data for each detected face
        """
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        landmarks_list = []
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                landmarks_list.append({
                    'landmarks': landmarks,
                    'num_landmarks': len(landmarks)
                })
        
        face_mesh.close()
        return landmarks_list