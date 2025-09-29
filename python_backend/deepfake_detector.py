"""
Deepfake Detector Module
Lightweight CNN ensemble approach using pure NumPy implementation
without heavy TensorFlow dependencies for disk space efficiency
"""

import cv2
import numpy as np
import pickle
import os
from typing import Tuple, List
import json
from lightweight_cnn import CNNEnsemble

class DeepfakeDetector:
    def __init__(self):
        """Initialize the lightweight CNN ensemble deepfake detector"""
        # Initialize CNN ensemble
        self.cnn_ensemble = CNNEnsemble()
        
        # Try to load pre-trained models
        self._load_models()
        
        # Fallback feature extractors for heuristic analysis
        self.feature_extractors = [
            'texture_features',
            'color_features', 
            'edge_features',
            'frequency_features'
        ]
    
    # Old ML models removed - now using CNN ensemble
    
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
        
        # Apply DCT (Discrete Cosine Transform)
        gray_float = gray.astype(np.float32)
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
        Predict if a face is real or fake using CNN ensemble
        
        Args:
            face: Face image as numpy array
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Use CNN ensemble for prediction
            prediction, confidence = self.cnn_ensemble.predict(face)
            return prediction, confidence
            
        except Exception as e:
            print(f"CNN prediction error: {e}, falling back to heuristic")
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
        if float(texture_entropy) > 4.5:  # High texture entropy might indicate artifacts
            score += 0.3
        
        if float(color_variance) < 0.01:  # Very low color variance might indicate processing
            score += 0.2
            
        if float(edge_density) > 0.3:  # High edge density might indicate compression artifacts
            score += 0.2
        
        # Simple threshold-based decision
        score_float = float(score) if hasattr(score, 'item') else score
        if score_float > 0.4:
            return 'FAKE', min(0.8, 0.5 + score_float)
        else:
            return 'REAL', min(0.8, 0.5 + (0.4 - score_float))
    
    def save_models(self, directory: str = 'cnn_models'):
        """Save CNN ensemble models to disk"""
        try:
            self.cnn_ensemble.save_models(directory)
            print(f"CNN ensemble saved to {directory}")
        except Exception as e:
            print(f"Error saving CNN models: {e}")
    
    def _load_models(self, directory: str = 'cnn_models'):
        """Load pre-trained CNN models from disk"""
        try:
            self.cnn_ensemble.load_models(directory)
            print("CNN ensemble models loaded successfully")
        except Exception as e:
            print(f"Error loading CNN models: {e}. Using default initialization.")
    
    def get_model_info(self) -> dict:
        """Get information about the current models"""
        cnn_info = self.cnn_ensemble.get_model_info()
        
        info = {
            'model_type': 'Lightweight CNN Ensemble',
            'ensemble_size': cnn_info['ensemble_size'],
            'architecture': cnn_info['architecture'],
            'input_size': cnn_info['input_size'],
            'model_names': cnn_info['model_names'],
            'trained_models': cnn_info['trained_models'],
            'fallback_features': self.feature_extractors
        }
        
        return info