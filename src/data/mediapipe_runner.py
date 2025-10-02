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
        import mediapipe as mp

        self.mp = mp
        self.spec = LandmarkSpec()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _extract_landmarks_from_result(self, result, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        out = np.full((self.spec.total_points, 3), np.nan, dtype=np.float32)
        idx = 0
        # face
        if result.face_landmarks is not None and result.face_landmarks.landmark:
            for lm in result.face_landmarks.landmark[: self.spec.face_landmarks]:
                out[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.spec.face_landmarks
        # pose
        if result.pose_landmarks is not None and result.pose_landmarks.landmark:
            for lm in result.pose_landmarks.landmark[: self.spec.pose_landmarks]:
                out[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.spec.pose_landmarks
        # left hand
        if result.left_hand_landmarks is not None and result.left_hand_landmarks.landmark:
            for lm in result.left_hand_landmarks.landmark[: self.spec.hand_landmarks]:
                out[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.spec.hand_landmarks
        # right hand
        if result.right_hand_landmarks is not None and result.right_hand_landmarks.landmark:
            for lm in result.right_hand_landmarks.landmark[: self.spec.hand_landmarks]:
                out[idx] = [lm.x, lm.y, lm.z]
                idx += 1
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
