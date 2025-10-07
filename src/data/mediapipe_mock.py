"""
Mock MediaPipe module for testing when MediaPipe is not available.
This provides a minimal interface that matches the expected MediaPipe Holistic API.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional


class MockHolistic:
    """Mock MediaPipe Holistic class."""
    
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 refine_face_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
    def process(self, image):
        """Mock process method that returns mock results."""
        height, width = image.shape[:2]
        
        # Create mock landmarks
        face_landmarks = self._create_mock_landmarks(468, width, height)  # Face
        pose_landmarks = self._create_mock_landmarks(33, width, height)   # Pose
        left_hand_landmarks = self._create_mock_landmarks(21, width, height)  # Left hand
        right_hand_landmarks = self._create_mock_landmarks(21, width, height)  # Right hand
        
        # Create mock results object
        class MockResults:
            def __init__(self, face, pose, left_hand, right_hand):
                self.face_landmarks = face
                self.pose_landmarks = pose
                self.left_hand_landmarks = left_hand
                self.right_hand_landmarks = right_hand
        
        return MockResults(face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks)
    
    def _create_mock_landmarks(self, num_landmarks, width, height):
        """Create mock landmark data."""
        landmarks = []
        for i in range(num_landmarks):
            # Create random but reasonable landmark positions
            x = np.random.uniform(0.1, 0.9) * width
            y = np.random.uniform(0.1, 0.9) * height
            z = np.random.uniform(-0.1, 0.1)  # Small depth variation
            
            landmark = type('Landmark', (), {
                'x': x,
                'y': y,
                'z': z,
                'visibility': np.random.uniform(0.5, 1.0)
            })()
            landmarks.append(landmark)
        
        return landmarks


def create_mock_holistic(**kwargs):
    """Create a mock holistic instance."""
    return MockHolistic(**kwargs)


# Mock the mediapipe module
class MockMediaPipe:
    def __init__(self):
        self.solutions = type('Solutions', (), {
            'holistic': type('Holistic', (), {
                'Holistic': MockHolistic
            })()
        })()
    
    def __getattr__(self, name):
        if name == 'solutions':
            return self.solutions
        return None


# Create mock module
mock_mediapipe = MockMediaPipe()
