"""Switch 2 left-arm demo with a simulated light.

The light starts ON. Real attention gating and device control are not connected.
"""

import argparse
from pathlib import Path
import time
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detector import Switch2Detector

# Pose landmark IDs for each shoulder, elbow, and wrist.
ARMS = {
    "LEFT": ((11, 13, 15), (255, 255, 0)),
    "RIGHT": ((12, 14, 16), (0, 165, 255)),
}
MIN_POINT_SCORE = 0.5  # Same confidence cutoff used by the detector.
DEFAULT_MODEL = Path(__file__).parent / "models" / "pose_landmarker_lite.task"


def point_score(point):
    """Check both scores before drawing the point."""
    return min(point.visibility or 0.0, point.presence or 0.0)


def draw_arms(frame, landmarks):
    """Draw the usable arm points so we can see what is being tracked."""
    height, width = frame.shape[:2]
    messages = []
    for side, (ids, color) in ARMS.items():
        pixels = {}
        scores = []
        for landmark_id, name in zip(ids, ("shoulder", "elbow", "wrist")):
            point = landmarks[landmark_id]
            score = point_score(point)
            scores.append(score)
            if score < MIN_POINT_SCORE or not (0 <= point.x < 1 and 0 <= point.y < 1):
                continue
            pixel = (int(point.x * width), int(point.y * height))
            pixels[landmark_id] = pixel
            cv2.circle(frame, pixel, 6, color, -1)
            label = f"{side[0]} {name} #{landmark_id}"
            cv2.putText(frame, label, pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, label, pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for start, end in zip(ids, ids[1:]):
            if start in pixels and end in pixels:
                cv2.line(frame, pixels[start], pixels[end], color, 3)
        status = "3/3 visible" if len(pixels) == 3 else f"{len(pixels)}/3 usable"
        messages.append(f"{side}: {status} | min score {min(scores):.2f}")
    return messages


def main():
    # Camera numbers can change when devices are reconnected.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--check", action="store_true", help="Check model inference without opening a camera")
    args = parser.parse_args()
    if sys.platform == "darwin" and int(mp.__version__.split(".")[0]) >= 1:
        parser.error("This Mac failed with MediaPipe 1.x. Use switch2/.venv/bin/python; see README.md.")
    if not args.model.is_file():
        parser.error(f"Model missing: {args.model}. See switch2/README.md for download instructions.")

    # Load the pretrained pose model. We are not training a model here.
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(options) as pose:
        if args.check:
            import numpy as np
            blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((256, 256, 3), dtype=np.uint8))
            result = pose.detect_for_video(blank, 0)
            print(f"Model inference passed. Poses in blank image: {len(result.pose_landmarks)}")
            return

        cap = cv2.VideoCapture(args.camera)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Camera {args.camera} did not open. Check permission or try another --camera number.")
            last_timestamp = -1
            last_status = None

            detector = Switch2Detector()
            light_on = True
            activation_count = 0

            # Testing only. Havish's real attention flag needs to replace this.
            test_attention_allowed = True

            print("Switch 2 simulated light demo. Keep the whole left arm visible. Press q to quit.", flush=True)
            while True:
                # Read the frame and change BGR to the RGB format MediaPipe expects.
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Camera {args.camera} opened but did not return a frame. Try another camera number.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(last_timestamp + 1, time.monotonic_ns() // 1_000_000)
                last_timestamp = timestamp

                # Get the body points and check whether the left-arm gesture finished.
                result = pose.detect_for_video(image, timestamp)
                points = (
                    result.pose_landmarks[0]
                    if result.pose_landmarks else None
                )

                triggered = detector.update(
                    points,
                    time.monotonic(),
                    attention_allowed=test_attention_allowed,
                )

                if triggered:
                    light_on = not light_on
                    activation_count += 1
                    print(
                        f"SIMULATED switch 2: "
                        f"{'ON' if light_on else 'OFF'} "
                        f"| activations: {activation_count}",
                        flush=True,
                    )

                messages = (
                    draw_arms(frame, points)
                    if points else ["NO POSE - show your upper body"]
                )

                lines = [
                    f"SIMULATED LIGHT: {'ON' if light_on else 'OFF'}"
                    f" | Activations: {activation_count}",
                    detector.status,
                    "Attention gate: TEST ENABLED | q: quit",
                    *messages,
                ]

                cv2.rectangle(
                    frame, (0, 0),
                    (frame.shape[1], 175),
                    (0, 0, 0), -1,
                )
                for row, text in enumerate(lines):
                    cv2.putText(frame, text, (12, 28 + row * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                status = tuple(message.split(" | ")[0] for message in messages)
                if status != last_status:
                    print("; ".join(messages), flush=True)
                    last_status = status

                # Not mirrored: L/R refer to the person's body sides.
                cv2.imshow("GAZE Switch 2 - Simulated Light", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # Always free the camera when we stop.
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
