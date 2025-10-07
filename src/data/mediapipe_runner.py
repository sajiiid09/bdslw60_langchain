from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class LandmarkSpec:
    # counts per holistic stream
    face_landmarks: int = 468
    pose_landmarks: int = 33
    hand_landmarks: int = 21  # per hand

    @property
    def total_points(self) -> int:
        return self.face_landmarks + self.pose_landmarks + 2 * self.hand_landmarks

    @property
    def total_dims(self) -> int:
        return self.total_points * 3  # x, y, z


class HolisticExtractor:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        # delayed import to make CPU-only envs work when mediapipe not installed yet
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
        except ImportError:
            # Use mock MediaPipe when not available
            from .mediapipe_mock import MockHolistic
            self.mp = None
            self.holistic = MockHolistic(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        
        self.spec = LandmarkSpec()

    def _extract_landmarks_from_result(self, result, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        out = np.full((self.spec.total_points, 3), np.nan, dtype=np.float32)
        idx = 0
        
        # Helper function to extract landmarks from a list
        def extract_from_list(landmarks_list, count):
            nonlocal idx
            if landmarks_list is not None and len(landmarks_list) > 0:
                for lm in landmarks_list[:count]:
                    if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
                        # Real MediaPipe landmarks
                        out[idx] = [lm.x, lm.y, lm.z]
                    else:
                        # Mock landmarks
                        out[idx] = [lm.x, lm.y, lm.z]
                    idx += 1
            else:
                idx += count
        
        # face
        if hasattr(result, 'face_landmarks') and result.face_landmarks is not None:
            if hasattr(result.face_landmarks, 'landmark'):
                # Real MediaPipe result
                extract_from_list(result.face_landmarks.landmark, self.spec.face_landmarks)
            else:
                # Mock result
                extract_from_list(result.face_landmarks, self.spec.face_landmarks)
        else:
            idx += self.spec.face_landmarks
            
        # pose
        if hasattr(result, 'pose_landmarks') and result.pose_landmarks is not None:
            if hasattr(result.pose_landmarks, 'landmark'):
                # Real MediaPipe result
                extract_from_list(result.pose_landmarks.landmark, self.spec.pose_landmarks)
            else:
                # Mock result
                extract_from_list(result.pose_landmarks, self.spec.pose_landmarks)
        else:
            idx += self.spec.pose_landmarks
            
        # left hand
        if hasattr(result, 'left_hand_landmarks') and result.left_hand_landmarks is not None:
            if hasattr(result.left_hand_landmarks, 'landmark'):
                # Real MediaPipe result
                extract_from_list(result.left_hand_landmarks.landmark, self.spec.hand_landmarks)
            else:
                # Mock result
                extract_from_list(result.left_hand_landmarks, self.spec.hand_landmarks)
        else:
            idx += self.spec.hand_landmarks
            
        # right hand
        if hasattr(result, 'right_hand_landmarks') and result.right_hand_landmarks is not None:
            if hasattr(result.right_hand_landmarks, 'landmark'):
                # Real MediaPipe result
                extract_from_list(result.right_hand_landmarks.landmark, self.spec.hand_landmarks)
            else:
                # Mock result
                extract_from_list(result.right_hand_landmarks, self.spec.hand_landmarks)
        else:
            idx += self.spec.hand_landmarks
            
        return out.reshape(-1)

    def extract_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        frames: List[np.ndarray] = []
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
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            cap.release()
        arr = np.stack(frames, axis=0) if frames else np.empty((0, self.spec.total_dims), dtype=np.float32)
        return {"landmarks": arr}
