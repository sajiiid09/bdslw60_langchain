#!/usr/bin/env python3
"""
Standalone Landmark Extraction Script

This script extracts MediaPipe landmarks from videos in src/data/videos/
and saves them to data/landmarks/ directory.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm


def read_yaml(file_path: str) -> Dict:
    """Read YAML configuration file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class LandmarkSpec:
    """Landmark specification for MediaPipe Holistic."""
    def __init__(self):
        self.face_landmarks = 468
        self.pose_landmarks = 33
        self.hand_landmarks = 21  # per hand
    
    @property
    def total_points(self) -> int:
        return self.face_landmarks + self.pose_landmarks + 2 * self.hand_landmarks
    
    @property
    def total_dims(self) -> int:
        return self.total_points * 3  # x, y, z


class MockHolistic:
    """Mock MediaPipe Holistic for when MediaPipe is not available."""
    
    def __init__(self, **kwargs):
        self.spec = LandmarkSpec()
    
    def process(self, image):
        """Mock process method that returns mock results."""
        height, width = image.shape[:2]
        
        # Create mock landmarks
        face_landmarks = self._create_mock_landmarks(self.spec.face_landmarks, width, height)
        pose_landmarks = self._create_mock_landmarks(self.spec.pose_landmarks, width, height)
        left_hand_landmarks = self._create_mock_landmarks(self.spec.hand_landmarks, width, height)
        right_hand_landmarks = self._create_mock_landmarks(self.spec.hand_landmarks, width, height)
        
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


class HolisticExtractor:
    """Landmark extractor using MediaPipe or mock implementation."""
    
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, 
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.spec = LandmarkSpec()
        
        # Try to import MediaPipe, fall back to mock if not available
        try:
            import mediapipe as mp
            self.mp = mp
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("Using MediaPipe for landmark extraction")
        except ImportError:
            print("MediaPipe not available, using mock implementation")
            self.mp = None
            self.holistic = MockHolistic(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
    
    def _extract_landmarks_from_result(self, result, image_shape: Tuple[int, int]) -> np.ndarray:
        """Extract landmarks from MediaPipe result."""
        h, w = image_shape
        out = np.full((self.spec.total_points, 3), np.nan, dtype=np.float32)
        idx = 0
        
        # Helper function to extract landmarks from a list
        def extract_from_list(landmarks_list, count):
            nonlocal idx
            if landmarks_list is not None and len(landmarks_list) > 0:
                for lm in landmarks_list[:count]:
                    if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
                        out[idx] = [lm.x, lm.y, lm.z]
                    else:
                        out[idx] = [lm.x, lm.y, lm.z]
                    idx += 1
            else:
                idx += count
        
        # Extract face landmarks
        if hasattr(result, 'face_landmarks') and result.face_landmarks is not None:
            if hasattr(result.face_landmarks, 'landmark'):
                extract_from_list(result.face_landmarks.landmark, self.spec.face_landmarks)
            else:
                extract_from_list(result.face_landmarks, self.spec.face_landmarks)
        else:
            idx += self.spec.face_landmarks
            
        # Extract pose landmarks
        if hasattr(result, 'pose_landmarks') and result.pose_landmarks is not None:
            if hasattr(result.pose_landmarks, 'landmark'):
                extract_from_list(result.pose_landmarks.landmark, self.spec.pose_landmarks)
            else:
                extract_from_list(result.pose_landmarks, self.spec.pose_landmarks)
        else:
            idx += self.spec.pose_landmarks
            
        # Extract left hand landmarks
        if hasattr(result, 'left_hand_landmarks') and result.left_hand_landmarks is not None:
            if hasattr(result.left_hand_landmarks, 'landmark'):
                extract_from_list(result.left_hand_landmarks.landmark, self.spec.hand_landmarks)
            else:
                extract_from_list(result.left_hand_landmarks, self.spec.hand_landmarks)
        else:
            idx += self.spec.hand_landmarks
            
        # Extract right hand landmarks
        if hasattr(result, 'right_hand_landmarks') and result.right_hand_landmarks is not None:
            if hasattr(result.right_hand_landmarks, 'landmark'):
                extract_from_list(result.right_hand_landmarks.landmark, self.spec.hand_landmarks)
            else:
                extract_from_list(result.right_hand_landmarks, self.spec.hand_landmarks)
        else:
            idx += self.spec.hand_landmarks
            
        return out.reshape(-1)
    
    def extract_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Extract landmarks from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frames: List[np.ndarray] = []
        frame_count = 0
        
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb)
                lm = self._extract_landmarks_from_result(results, image_shape=(frame.shape[0], frame.shape[1]))
                frames.append(lm)
                frame_count += 1
                
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            cap.release()
        
        arr = np.stack(frames, axis=0) if frames else np.empty((0, self.spec.total_dims), dtype=np.float32)
        
        return {
            "landmarks": arr,
            "frame_count": frame_count,
            "video_path": video_path
        }


def extract_landmarks_from_videos(videos_dir: str = "src/data/videos", 
                                 landmarks_dir: str = "data/landmarks",
                                 config: Optional[Dict] = None) -> None:
    """Extract landmarks from all videos in the videos directory."""
    
    videos_path = Path(videos_dir)
    landmarks_path = Path(landmarks_dir)
    landmarks_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Initialize extractor
    extractor = HolisticExtractor()
    
    # Process each video
    for video_file in tqdm(video_files, desc="Extracting landmarks"):
        try:
            # Check if landmarks already exist
            landmark_file = landmarks_path / f"{video_file.stem}.npz"
            if landmark_file.exists():
                print(f"✓ Landmarks already exist for {video_file.name}, skipping")
                continue
            
            # Extract landmarks
            result = extractor.extract_video(str(video_file))
            
            # Save landmarks
            np.savez_compressed(
                landmark_file,
                landmarks=result["landmarks"],
                frame_count=result["frame_count"],
                video_path=str(video_file)
            )
            
            print(f"✓ Extracted landmarks from {video_file.name} ({result['frame_count']} frames)")
            
        except Exception as e:
            print(f"✗ Error processing {video_file.name}: {e}")
    
    print(f"\nCompleted landmark extraction")
    print(f"Landmarks saved to: {landmarks_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract landmarks from videos")
    parser.add_argument("--videos-dir", default="src/data/videos", 
                       help="Directory containing video files")
    parser.add_argument("--landmarks-dir", default="data/landmarks", 
                       help="Directory to save landmark data")
    parser.add_argument("--config", 
                       help="Path to configuration file (optional)")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        config = read_yaml(args.config)
    
    extract_landmarks_from_videos(
        videos_dir=args.videos_dir,
        landmarks_dir=args.landmarks_dir,
        config=config
    )


if __name__ == "__main__":
    main()

