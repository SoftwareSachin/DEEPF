"""
Deepfake Detector Module
Lightweight ensemble approach using traditional ML techniques
without heavy TensorFlow dependencies for disk space efficiency
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
from typing import Tuple, List
import json

class DeepfakeDetector:
    def __init__(self):
        """Initialize the ensemble deepfake detector"""
        self.models = {}
        self.scalers = {}
        self.feature_extractors = [
            'texture_features',
            'color_features', 
            'edge_features',
            'frequency_features'
        ]
        
        # Initialize models
        self._initialize_models()
        
        # Try to load pre-trained models
        self._load_models()
    
    def _initialize_models(self):
        """Initialize the ensemble of ML models"""
        # Random Forest for texture analysis
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # SVM for edge pattern analysis  
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['rf']),
                ('svm', self.models['svm'])
            ],
            voting='soft'
        )
        
        # Initialize scalers for each feature type
        for feature_type in self.feature_extractors:
            self.scalers[feature_type] = StandardScaler()
    
    def extract_texture_features(self, face: np.ndarray) -> np.ndarray:
        """Extract texture-based features using Local Binary Patterns"""
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
        
        # Local Binary Pattern
        def local_binary_pattern(img, radius=1, n_points=8):
            """Simplified LBP implementation"""
            h, w = img.shape
            lbp = np.zeros((h, w), dtype=np.uint8)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = img[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < h and 0 <= y < w:
                            if img[x, y] >= center:
                                code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        
        lbp = local_binary_pattern(gray)
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)  # Normalize
        
        # Additional texture features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Combine features
        features = np.concatenate([hist, [mean_val, std_val]])
        return features
    
    def extract_color_features(self, face: np.ndarray) -> np.ndarray:
        """Extract color distribution features"""
        if len(face.shape) != 3:
            # Convert grayscale to BGR
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        
        features = []
        
        # Color histograms for each channel
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([face], [i], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            features.extend(hist)
        
        # Color moments
        for i in range(3):
            channel = face[:, :, i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            skewness = np.mean(((channel - mean_val) / (std_val + 1e-6)) ** 3)
            features.extend([mean_val, std_val, skewness])
        
        return np.array(features, dtype=np.float32)
    
    def extract_edge_features(self, face: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Canny edges
        canny = cv2.Canny(gray, 50, 150)
        
        # Edge statistics
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.sum(canny > 0) / canny.size,  # Edge density
            np.mean(sobel_x),
            np.std(sobel_x),
            np.mean(sobel_y), 
            np.std(sobel_y)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_frequency_features(self, face: np.ndarray) -> np.ndarray:
        """Extract frequency domain features using DCT"""
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face
        
        # Resize to standard size for consistent features
        gray = cv2.resize(gray, (64, 64))
        
        # Apply DCT
        gray_float = np.float32(gray)
        dct = cv2.dct(gray_float)
        
        # Extract features from DCT coefficients
        # Take low-frequency components (top-left 8x8 block)
        low_freq = dct[:8, :8].flatten()
        
        # Statistical features
        features = [
            np.mean(dct),
            np.std(dct),
            np.mean(low_freq),
            np.std(low_freq),
            np.sum(np.abs(dct)),  # Total energy
        ]
        
        # Add some DCT coefficients as features
        features.extend(low_freq[:20])  # First 20 DCT coefficients
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_features(self, face: np.ndarray) -> np.ndarray:
        """Extract all feature types and combine them"""
        features = []
        
        # Extract each feature type
        texture_feat = self.extract_texture_features(face)
        color_feat = self.extract_color_features(face)
        edge_feat = self.extract_edge_features(face)
        freq_feat = self.extract_frequency_features(face)
        
        # Combine all features
        all_features = np.concatenate([texture_feat, color_feat, edge_feat, freq_feat])
        
        return all_features
    
    def predict(self, face: np.ndarray) -> Tuple[str, float]:
        """
        Predict if a face is real or fake
        
        Args:
            face: Face image as numpy array
        
        Returns:
            Tuple of (prediction, confidence)
        """
        # Extract features
        features = self.extract_all_features(face)
        features = features.reshape(1, -1)
        
        # If models are not trained, use heuristic approach
        if not hasattr(self.ensemble, 'estimators_'):
            return self._heuristic_prediction(face)
        
        # Use trained ensemble
        try:
            prediction = self.ensemble.predict(features)[0]
            probabilities = self.ensemble.predict_proba(features)[0]
            confidence = max(probabilities)
            
            result = 'FAKE' if prediction == 1 else 'REAL'
            return result, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._heuristic_prediction(face)
    
    def _heuristic_prediction(self, face: np.ndarray) -> Tuple[str, float]:
        """
        Fallback heuristic-based prediction when models aren't trained
        Uses statistical analysis of image properties
        """
        features = self.extract_all_features(face)
        
        # Simple heuristic scoring based on feature analysis
        score = 0.0
        confidence = 0.6  # Default confidence for heuristic
        
        # Analyze texture uniformity (deepfakes often have unusual texture patterns)
        texture_features = self.extract_texture_features(face)
        texture_entropy = -np.sum(texture_features[:-2] * np.log(texture_features[:-2] + 1e-10))
        
        # Analyze color distribution
        color_features = self.extract_color_features(face)
        color_variance = np.var(color_features)
        
        # Analyze edge consistency
        edge_features = self.extract_edge_features(face)
        edge_density = edge_features[2]  # Edge density feature
        
        # Scoring heuristics
        if texture_entropy > 4.5:  # High texture entropy might indicate artifacts
            score += 0.3
        
        if color_variance < 0.01:  # Very low color variance might indicate processing
            score += 0.2
            
        if edge_density > 0.3:  # High edge density might indicate compression artifacts
            score += 0.2
        
        # Simple threshold-based decision
        if score > 0.4:
            return 'FAKE', min(0.8, 0.5 + score)
        else:
            return 'REAL', min(0.8, 0.5 + (0.4 - score))
    
    def train_on_batch(self, faces: List[np.ndarray], labels: List[int]):
        """
        Train the models on a batch of faces
        
        Args:
            faces: List of face images
            labels: List of labels (0 for real, 1 for fake)
        """
        if len(faces) != len(labels):
            raise ValueError("Number of faces and labels must match")
        
        # Extract features for all faces
        all_features = []
        for face in faces:
            features = self.extract_all_features(face)
            all_features.append(features)
        
        X = np.array(all_features)
        y = np.array(labels)
        
        # Train the ensemble
        self.ensemble.fit(X, y)
        
        print(f"Trained on {len(faces)} faces")
    
    def save_models(self, directory: str = 'models'):
        """Save trained models to disk"""
        os.makedirs(directory, exist_ok=True)
        
        if hasattr(self.ensemble, 'estimators_'):
            with open(os.path.join(directory, 'ensemble.pkl'), 'wb') as f:
                pickle.dump(self.ensemble, f)
            
            # Save scalers
            for name, scaler in self.scalers.items():
                with open(os.path.join(directory, f'scaler_{name}.pkl'), 'wb') as f:
                    pickle.dump(scaler, f)
            
            print(f"Models saved to {directory}")
    
    def _load_models(self, directory: str = 'models'):
        """Load pre-trained models from disk"""
        if not os.path.exists(directory):
            print("No pre-trained models found. Using heuristic detection.")
            return
        
        try:
            ensemble_path = os.path.join(directory, 'ensemble.pkl')
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    self.ensemble = pickle.load(f)
                
                # Load scalers
                for name in self.feature_extractors:
                    scaler_path = os.path.join(directory, f'scaler_{name}.pkl')
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[name] = pickle.load(f)
                
                print("Pre-trained models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}. Using heuristic detection.")
    
    def get_model_info(self) -> dict:
        """Get information about the current models"""
        info = {
            'ensemble_trained': hasattr(self.ensemble, 'estimators_'),
            'feature_extractors': self.feature_extractors,
            'models': list(self.models.keys())
        }
        
        if info['ensemble_trained']:
            info['model_details'] = {
                'n_estimators': len(self.ensemble.estimators_),
                'feature_count': len(self.feature_extractors)
            }
        
        return info