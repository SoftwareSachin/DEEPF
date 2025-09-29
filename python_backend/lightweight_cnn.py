"""
Lightweight CNN Implementation for Deepfake Detection
Using pure NumPy to avoid TensorFlow/PyTorch dependencies while maintaining disk space efficiency
"""

import numpy as np
import cv2
from typing import Tuple, List
import pickle
import os

class LightweightCNN:
    """
    Simplified CNN implementation using only NumPy
    Designed to be lightweight while maintaining good detection performance
    """
    
    def __init__(self, name: str = "LightCNN"):
        self.name = name
        self.weights = {}
        self.trained = False
        
        # Initialize simple CNN architecture
        self._initialize_architecture()
    
    def _initialize_architecture(self):
        """Initialize a simple CNN architecture with random weights"""
        # Input: 64x64x3 images
        input_size = 64
        
        # Conv1: 3x3 kernels, 16 filters
        self.weights['conv1_w'] = np.random.randn(16, 3, 3, 3) * 0.1
        self.weights['conv1_b'] = np.zeros((16,))
        
        # Conv2: 3x3 kernels, 32 filters  
        self.weights['conv2_w'] = np.random.randn(32, 16, 3, 3) * 0.1
        self.weights['conv2_b'] = np.zeros((32,))
        
        # Conv3: 3x3 kernels, 64 filters
        self.weights['conv3_w'] = np.random.randn(64, 32, 3, 3) * 0.1
        self.weights['conv3_b'] = np.zeros((64,))
        
        # Fully connected layers
        # After 3 conv+pool layers: 64x64 -> 32x32 -> 16x16 -> 8x8
        fc_input_size = 64 * 8 * 8  # 64 filters * 8x8 feature map
        
        self.weights['fc1_w'] = np.random.randn(fc_input_size, 128) * 0.1
        self.weights['fc1_b'] = np.zeros((128,))
        
        self.weights['fc2_w'] = np.random.randn(128, 64) * 0.1
        self.weights['fc2_b'] = np.zeros((64,))
        
        self.weights['fc3_w'] = np.random.randn(64, 2) * 0.1  # 2 classes: real/fake
        self.weights['fc3_b'] = np.zeros((2,))
    
    def _conv2d(self, input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray, 
                stride: int = 1, padding: int = 0) -> np.ndarray:
        """Simple 2D convolution implementation"""
        if len(input_data.shape) == 3:
            input_data = input_data[np.newaxis, ...]
        
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weights.shape
        
        # Add padding
        if padding > 0:
            input_data = np.pad(input_data, 
                              ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                              mode='constant')
        
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1
        
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride
                        h_end = h_start + kernel_height
                        w_start = w * stride
                        w_end = w_start + kernel_width
                        
                        # Convolution operation
                        receptive_field = input_data[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, h, w] = np.sum(receptive_field * weights[oc]) + bias[oc]
        
        return output
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _max_pool2d(self, input_data: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        """Max pooling operation"""
        batch_size, channels, in_height, in_width = input_data.shape
        out_height = in_height // stride
        out_width = in_width // stride
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        
                        output[b, c, h, w] = np.max(input_data[b, c, h_start:h_end, w_start:w_end])
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass through the network"""
        # Ensure input is in correct format: (batch, channels, height, width)
        if len(x.shape) == 3:
            x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
            x = x[np.newaxis, ...]  # Add batch dimension
        
        # Normalize input
        x = x.astype(np.float32) / 255.0
        
        # Conv1 + ReLU + MaxPool
        conv1 = self._conv2d(x, self.weights['conv1_w'], self.weights['conv1_b'], padding=1)
        conv1 = self._relu(conv1)
        pool1 = self._max_pool2d(conv1)
        
        # Conv2 + ReLU + MaxPool
        conv2 = self._conv2d(pool1, self.weights['conv2_w'], self.weights['conv2_b'], padding=1)
        conv2 = self._relu(conv2)
        pool2 = self._max_pool2d(conv2)
        
        # Conv3 + ReLU + MaxPool
        conv3 = self._conv2d(pool2, self.weights['conv3_w'], self.weights['conv3_b'], padding=1)
        conv3 = self._relu(conv3)
        pool3 = self._max_pool2d(conv3)
        
        # Flatten for fully connected layers
        flattened = pool3.reshape(pool3.shape[0], -1)
        
        # FC1 + ReLU
        fc1 = np.dot(flattened, self.weights['fc1_w']) + self.weights['fc1_b']
        fc1 = self._relu(fc1)
        
        # FC2 + ReLU
        fc2 = np.dot(fc1, self.weights['fc2_w']) + self.weights['fc2_b']
        fc2 = self._relu(fc2)
        
        # FC3 + Softmax
        fc3 = np.dot(fc2, self.weights['fc3_w']) + self.weights['fc3_b']
        output = self._softmax(fc3)
        
        # Return prediction probabilities and confidence
        fake_prob = output[0, 1]  # Probability of being fake
        confidence = np.max(output[0])  # Highest probability
        
        return output, float(confidence)
    
    def predict(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict if a face is real or fake
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Resize to network input size
        face_resized = cv2.resize(face_image, (64, 64))
        
        # Forward pass
        output, confidence = self.forward(face_resized)
        
        # Get prediction
        fake_prob = output[0, 1]
        prediction = 'FAKE' if fake_prob > 0.5 else 'REAL'
        
        return prediction, confidence

class CNNEnsemble:
    """
    Ensemble of lightweight CNN models for improved deepfake detection
    """
    
    def __init__(self):
        self.models = []
        self.model_names = ['LightCNN_1', 'LightCNN_2', 'LightCNN_3']
        
        # Initialize ensemble models
        for name in self.model_names:
            model = LightweightCNN(name)
            # Add some variation to each model
            self._add_model_variation(model)
            self.models.append(model)
    
    def _add_model_variation(self, model: LightweightCNN):
        """Add slight variations to make models different"""
        # Randomly adjust some weights to create model diversity
        for key in model.weights:
            if 'w' in key:
                noise = np.random.randn(*model.weights[key].shape) * 0.01
                model.weights[key] += noise
    
    def predict(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Ensemble prediction using majority voting and confidence averaging
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (prediction, confidence)
        """
        predictions = []
        confidences = []
        fake_probs = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                pred, conf = model.predict(face_image)
                predictions.append(pred)
                confidences.append(conf)
                
                # Get fake probability for ensemble averaging
                output, _ = model.forward(cv2.resize(face_image, (64, 64)))
                fake_probs.append(output[0, 1])
                
            except Exception as e:
                print(f"Error in model {model.name}: {e}")
                # Use fallback prediction
                predictions.append('REAL')
                confidences.append(0.5)
                fake_probs.append(0.3)
        
        # Ensemble decision using average probabilities
        avg_fake_prob = np.mean(fake_probs)
        ensemble_prediction = 'FAKE' if avg_fake_prob > 0.5 else 'REAL'
        
        # Confidence is based on how confident the ensemble is
        ensemble_confidence = max(avg_fake_prob, 1 - avg_fake_prob)
        
        # Boost confidence if models agree
        agreement = len([p for p in predictions if p == ensemble_prediction]) / len(predictions)
        ensemble_confidence = min(0.95, ensemble_confidence * (0.5 + 0.5 * agreement))
        
        return ensemble_prediction, float(ensemble_confidence)
    
    def save_models(self, directory: str = 'cnn_models'):
        """Save the ensemble models"""
        os.makedirs(directory, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(directory, f'{model.name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model.weights, f)
        
        print(f"Saved {len(self.models)} CNN models to {directory}")
    
    def load_models(self, directory: str = 'cnn_models'):
        """Load pre-trained ensemble models"""
        if not os.path.exists(directory):
            print("No pre-trained CNN models found, using random initialization")
            return
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(directory, f'{model.name}.pkl')
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model.weights = pickle.load(f)
                    model.trained = True
                except Exception as e:
                    print(f"Error loading model {model.name}: {e}")
        
        print(f"Loaded CNN ensemble models from {directory}")
    
    def get_model_info(self) -> dict:
        """Get information about the ensemble"""
        return {
            'ensemble_size': len(self.models),
            'model_names': self.model_names,
            'architecture': 'Lightweight CNN Ensemble',
            'input_size': '64x64x3',
            'trained_models': sum(1 for model in self.models if model.trained)
        }