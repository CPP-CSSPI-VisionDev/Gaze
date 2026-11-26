"""Attention-gated whole-arm gesture task.

Run this file from the repository root with:
    python vision/gestures/gesture_task.py

The detector uses MediaPipe Pose rather than hand landmarks so a small wrist or
finger movement does not count as the gesture.
"""

from collections import deque
import time

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
HISTORY_SECONDS = 0.55
TRIGGER_DISTANCE_IN_SHOULDER_WIDTHS = 1.4
MIN_HORIZONTAL_DISPLACEMENT_IN_SHOULDER_WIDTHS = 1.0
GREEN_SECONDS = 1.0

# returns a normal distance between to pose landmarks (i.e. arms and wrists)
def normalized_distance(first, second):
    """Return the distance between two normalized Pose landmarks."""
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def main():
    # Sets up the poses and the video capture
    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # this runs if the camera did not open for some reason
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    # keeps a log of the wrist history and provides the time until green and cooldown
    wrist_history = deque()
    green_until = 0.0
    cooldown_until = 0.0

    # checks the pose to see if the image is clear enough and is detected well enough for the circle to turn green
    with pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:   # Reads the frame while it the pose and image are clear enough
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # sets the frame and the length of time the dot shows the color
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            now = time.monotonic()
            gesture_detected = False

            # checks the distance and landmarks of where the person's body is and sets it 
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[pose_module.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[pose_module.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[pose_module.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[pose_module.PoseLandmark.RIGHT_WRIST]

                shoulder_width = normalized_distance(left_shoulder, right_shoulder)
                visible_shoulders = min(left_shoulder.visibility, right_shoulder.visibility)
                visible_left_wrist = left_wrist.visibility
                visible_right_wrist = right_wrist.visibility

                # checks shoulders are there and the shoulders are there to see which wrist is seen
                if shoulder_width > 0 and visible_shoulders >= 0.5:
                    visible_wrists = []
                    if visible_left_wrist >= 0.5:
                        visible_wrists.append(("left", left_wrist))
                    if visible_right_wrist >= 0.5:
                        visible_wrists.append(("right", right_wrist))

                    # Keep the recent wrist positions for both arms. A large
                    # displacement within this short window is the gesture.
                    wrist_history.extend((side, wrist.x, wrist.y, now) for side, wrist in visible_wrists)
                    while wrist_history and now - wrist_history[0][3] > HISTORY_SECONDS:
                        wrist_history.popleft()

                    for side, wrist in visible_wrists:
                        previous = [
                            (x, y)
                            for old_side, x, y, timestamp in wrist_history
                            if old_side == side and now - timestamp >= 0.20
                        ]
                        if previous:
                            max_travel = max(
                                ((wrist.x - x) ** 2 + (wrist.y - y) ** 2) ** 0.5
                                for x, y in previous
                            )
                            max_horizontal_travel = max(
                                abs(wrist.x - x)
                                for x, _ in previous
                            )
                            if (
                                max_travel >= shoulder_width * TRIGGER_DISTANCE_IN_SHOULDER_WIDTHS
                                and max_horizontal_travel
                                >= shoulder_width * MIN_HORIZONTAL_DISPLACEMENT_IN_SHOULDER_WIDTHS
                            ):
                                gesture_detected = True
                                break
                
                # checks if gesutre is detected and long enough, where it will set the length of time for the green dot to show
                if gesture_detected and now >= cooldown_until:
                    green_until = now + GREEN_SECONDS
                    cooldown_until = green_until
                    wrist_history.clear()

                # draws the landmarks seen like face, wrist, and arms/shoulders
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    pose_module.POSE_CONNECTIONS,
                )

            # sets the circle to green if the whole arm pose registers, and otherwise it is red
            circle_color = (0, 255, 0) if now < green_until else (0, 0, 255)
            cv2.circle(frame, (60, 60), 40, circle_color, -1)
            cv2.putText(
                frame,
                "Whole-arm gesture task | q = quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Gesture Task", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # finishes gesture and stops recording when ended/closed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()