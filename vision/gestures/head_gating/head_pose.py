"""Head orientation in OpenCV camera coordinates: x right, y down, z away."""
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np

_data = json.loads(Path(__file__).with_name('canonical_landmarks.json').read_text())
LANDMARK_IDS = tuple(_data['ids'])
# MediaPipe canonical coordinates are x right, y up, z toward the viewer (cm).
# Rotate the model 180 degrees about x, NOT the estimated Euler angles.
MODEL_POINTS = np.asarray(_data['points'], dtype=np.float64) * [1, -1, -1]
MODEL_POINTS -= (MODEL_POINTS[LANDMARK_IDS.index(33)] + MODEL_POINTS[LANDMARK_IDS.index(263)]) / 2
FORWARD = np.array([0., 0., -1.])
FACING = 'FACING CAMERA'
NOT_FACING = 'NOT FACING'
UNKNOWN = 'UNKNOWN'


@dataclass
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int
    calibrated: bool = False

    @classmethod
    def load(cls, path, width, height):
        if path is None:
            return cls(np.array([[width, 0., width / 2], [0., width, height / 2], [0., 0., 1.]]), np.zeros(5), width, height)
        data = json.loads(Path(path).read_text())
        w, h = data['image_size']
        if abs(width / height - w / h) > 0.01:
            raise ValueError('Calibration aspect ratio differs from video; recalibrate at this resolution.')
        k = np.array(data['camera_matrix'], dtype=float)
        d = np.array(data['distortion'], dtype=float)
        if k.shape != (3, 3) or d.size not in (4, 5, 8, 12, 14) or not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError('Invalid camera calibration.')
        k[0] *= width / w
        k[1] *= height / h
        return cls(k, d, width, height, True)


@dataclass
class HeadPose:
    rotation: np.ndarray
    translation: np.ndarray
    forward: np.ndarray
    to_camera: np.ndarray
    angular_error: float
    yaw: float
    pitch: float
    reprojection_error: float


def angle_between(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denom) or denom < 1e-9:
        raise ValueError('Invalid direction vector')
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1., 1.))))


def pose_from_transform(rotation, translation, reprojection_error=0.):
    t = np.asarray(translation, dtype=float).reshape(3)
    forward = rotation @ FORWARD
    to_camera = -t / np.linalg.norm(t)
    error = angle_between(forward, to_camera)
    # Direction angles are for display only. Classification uses the full vectors.
    yaw = float(np.degrees(np.arctan2(forward[0], -forward[2])))
    pitch = float(np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2]))))
    return HeadPose(rotation, t, forward, to_camera, error, yaw, pitch, reprojection_error)


def estimate_from_points(image_points, camera, face_width, max_error_ratio=0.03):
    points = np.asarray(image_points, dtype=np.float64)
    if points.shape != (len(MODEL_POINTS), 2) or not np.isfinite(points).all():
        return None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            MODEL_POINTS, points, camera.matrix, camera.distortion,
            iterationsCount=100, reprojectionError=max(2., face_width * max_error_ratio),
            confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < int(np.ceil(len(points) * 0.8)):
            return None
        indices = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(MODEL_POINTS[indices], points[indices], camera.matrix, camera.distortion, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (rotation @ MODEL_POINTS.T + tvec)[2]
        if np.any(depths <= 0) or not np.isfinite(tvec).all():
            return None
        projected, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, camera.matrix, camera.distortion)
        # Check ALL correspondences: do not silently accept an occluded half-face.
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - points) ** 2, axis=1))))
        if error > max(2., face_width * max_error_ratio):
            return None
        return pose_from_transform(rotation, tvec, error)
    except (cv2.error, ValueError):
        return None


def estimate_head_pose(landmarks, camera, face_width, max_error_ratio=0.03):
    points = np.array([(landmarks[i].x * camera.width, landmarks[i].y * camera.height) for i in LANDMARK_IDS])
    return estimate_from_points(points, camera, face_width, max_error_ratio)


def is_looking(pose, was_looking=False, enter_threshold_degrees=12., exit_threshold_degrees=18.):
    if pose is None or not np.isfinite(pose.angular_error):
        return False
    return pose.angular_error <= (exit_threshold_degrees if was_looking else enter_threshold_degrees)


class AttentionGate:
    """EMA angular error + hysteresis + consecutive frames; invalid => UNKNOWN."""
    def __init__(self, enter=12., exit=18., consecutive=3, alpha=0.5):
        if not 0 < enter < exit < 90 or consecutive < 1 or not 0 < alpha <= 1:
            raise ValueError('Require 0 < enter < exit < 90, frames >= 1, and 0 < alpha <= 1.')
        self.enter, self.exit, self.consecutive, self.alpha = enter, exit, consecutive, alpha
        self.reset()

    def reset(self):
        self.status, self.smoothed, self.pending, self.count = UNKNOWN, None, None, 0

    def update(self, pose):
        if pose is None or not np.isfinite(pose.angular_error):
            self.reset()
            return self.status
        error = pose.angular_error
        self.smoothed = error if self.smoothed is None else self.alpha * error + (1 - self.alpha) * self.smoothed
        threshold = self.exit if self.status == FACING else self.enter
        # Both raw and smoothed values must agree to enter green.
        candidate = FACING if max(error, self.smoothed) <= threshold else NOT_FACING
        if candidate == self.status:
            self.pending, self.count = None, 0
        else:
            self.count = self.count + 1 if candidate == self.pending else 1
            self.pending = candidate
            if self.count >= self.consecutive:
                self.status, self.pending, self.count = candidate, None, 0
        return self.status
