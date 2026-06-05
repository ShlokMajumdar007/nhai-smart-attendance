"""
face_detector.py
MediaPipe-based face detection with alignment, quality checks,
and preprocessing for MobileFaceNet input.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for face alignment
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    landmarks: np.ndarray              # (468, 3) full mesh
    aligned_face: np.ndarray           # 112x112 aligned crop
    confidence: float
    blur_score: float
    brightness: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class FaceDetector:
    """
    Wraps MediaPipe Face Mesh for detection + alignment.
    Performs quality checks before returning a detection.
    """

    def __init__(
        self,
        min_face_size: int = 80,
        blur_threshold: float = 100.0,
        brightness_min: int = 40,
        brightness_max: int = 220,
        target_size: Tuple[int, int] = (112, 112),
        min_detection_confidence: float = 0.7,
        static_image_mode: bool = False,
    ):
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.target_size = target_size

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=2,           # detect up to 2 to reject multi-face
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Main entry point. Returns a FaceDetection or None if no valid face.
        Rejects: multiple faces, too small, blurry, bad brightness.
        """
        if frame is None or frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # --- Step 1: Face mesh ---
        mesh_result = self.face_mesh.process(rgb)

        if not mesh_result.multi_face_landmarks:
            return None

        # Reject multiple faces
        if len(mesh_result.multi_face_landmarks) > 1:
            return FaceDetection(
                bbox=(0, 0, 0, 0),
                landmarks=np.array([]),
                aligned_face=np.array([]),
                confidence=0.0,
                blur_score=0.0,
                brightness=0.0,
                is_valid=False,
                rejection_reason="multiple_faces",
            )

        face_landmarks = mesh_result.multi_face_landmarks[0]

        # Convert normalized landmarks to pixel coords
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # --- Step 2: Bounding box ---
        xs = landmarks_px[:, 0]
        ys = landmarks_px[:, 1]
        x1, y1 = int(np.min(xs)), int(np.min(ys))
        x2, y2 = int(np.max(xs)), int(np.max(ys))
        face_w, face_h = x2 - x1, y2 - y1

        # Reject too-small faces
        if face_w < self.min_face_size or face_h < self.min_face_size:
            return FaceDetection(
                bbox=(x1, y1, face_w, face_h),
                landmarks=landmarks_px,
                aligned_face=np.array([]),
                confidence=0.0,
                blur_score=0.0,
                brightness=0.0,
                is_valid=False,
                rejection_reason="face_too_small",
            )

        # --- Step 3: Crop for quality checks ---
        pad = 20
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        blur_score = self._compute_blur(face_crop)
        brightness = self._compute_brightness(face_crop)

        if blur_score < self.blur_threshold:
            return FaceDetection(
                bbox=(x1, y1, face_w, face_h),
                landmarks=landmarks_px,
                aligned_face=np.array([]),
                confidence=0.0,
                blur_score=blur_score,
                brightness=brightness,
                is_valid=False,
                rejection_reason="blurry",
            )

        if brightness < self.brightness_min or brightness > self.brightness_max:
            return FaceDetection(
                bbox=(x1, y1, face_w, face_h),
                landmarks=landmarks_px,
                aligned_face=np.array([]),
                confidence=0.0,
                blur_score=blur_score,
                brightness=brightness,
                is_valid=False,
                rejection_reason="bad_brightness",
            )

        # --- Step 4: Align face ---
        aligned = self._align_face(frame, landmarks_px)
        if aligned is None:
            return None

        # Get detection confidence from face_detection module
        det_result = self.face_detection.process(rgb)
        confidence = 0.9  # default; override if detection available
        if det_result.detections:
            confidence = det_result.detections[0].score[0]

        return FaceDetection(
            bbox=(x1, y1, face_w, face_h),
            landmarks=landmarks_px,
            aligned_face=aligned,
            confidence=confidence,
            blur_score=blur_score,
            brightness=brightness,
            is_valid=True,
        )

    def _align_face(
        self, frame: np.ndarray, landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Align face using eye centers.
        Rotates image so eyes are horizontal, then crops to target_size.
        """
        try:
            left_eye = landmarks[LEFT_EYE_IDX, :2].mean(axis=0)
            right_eye = landmarks[RIGHT_EYE_IDX, :2].mean(axis=0)

            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            eye_center = ((left_eye + right_eye) / 2).astype(np.float32)
            M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)

            rotated = cv2.warpAffine(
                frame, M, (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
            )

            # Recompute landmarks after rotation
            ones = np.ones((landmarks.shape[0], 1))
            lm_homo = np.hstack([landmarks[:, :2], ones])
            rotated_lm = (M @ lm_homo.T).T

            xs = rotated_lm[:, 0]
            ys = rotated_lm[:, 1]
            x1, y1 = int(np.min(xs)), int(np.min(ys))
            x2, y2 = int(np.max(xs)), int(np.max(ys))

            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            h, w = rotated.shape[:2]
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            face_crop = rotated[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            aligned = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
            aligned = self._preprocess(aligned)
            return aligned

        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return None

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization and brightness normalization.
        Returns float32 normalized to [-1, 1].
        """
        # Convert to LAB for CLAHE on luminance channel only
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Normalize to [-1, 1] for MobileFaceNet
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        return face

    @staticmethod
    def _compute_blur(face: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_brightness(face: np.ndarray) -> float:
        """Mean brightness of V channel in HSV."""
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    def get_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple) -> dict:
        """
        Estimate yaw/pitch/roll from facial landmarks.
        Used by liveness detection for head-turn challenge.
        Returns dict with yaw, pitch, roll in degrees.
        """
        h, w = frame_shape[:2]

        # 3D reference points (standard face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ], dtype=np.float64)

        # Corresponding 2D image points from landmarks
        image_points = np.array([
            landmarks[NOSE_TIP_IDX, :2],
            landmarks[152, :2],      # Chin
            landmarks[LEFT_EYE_IDX[0], :2],
            landmarks[RIGHT_EYE_IDX[0], :2],
            landmarks[LEFT_MOUTH_IDX, :2],
            landmarks[RIGHT_MOUTH_IDX, :2],
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return {
            "yaw": float(angles[1]),
            "pitch": float(angles[0]),
            "roll": float(angles[2]),
        }

    def close(self):
        self.face_mesh.close()
        self.face_detection.close()