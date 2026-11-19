# Switch 2 left arm gesture

This is my part for switch 2. Tejas asked me to use my left arm because it
will be closer to the camera. The demo is working locally, but it is still
using a simulated light. It does not control the real switch yet.

## How the gesture works

1. Start with the left arm down for about 0.3 seconds. Wait for READY.
2. Raise the left arm above shoulder level, with the elbow near shoulder height.
3. Hold it for one second. That gives one activation.
4. Lower the arm again before doing the next gesture.

The light starts ON. The first activation turns it OFF, the second turns it
ON, and it keeps following that pattern. Holding the arm up does not keep
switching it. Restarting the demo starts the simulated light ON again.

The whole left shoulder, elbow, and wrist need to stay visible. I first
had a problem where the light stayed ON and the count stayed at zero.
The screen said LEFT ARM TRACKING UNCERTAIN. After I moved back so the
whole arm was visible, it worked.

## Files

- `detector.py`: checks the left arm and returns True once for each gesture.
- `demo.py`: opens the camera, draws the landmarks, and toggles the simulated light.
- `requirements.txt`: the versions used for this standalone demo on my Mac.
- `TESTING.md`: what I tested, what passed, and what still needs checking.

## Setup on Mac

Run these from the Gaze repository root, not from inside the switch2 folder.
Use Python 3.11.

```bash
python3.11 -m venv vision/gestures/switch2/.venv
vision/gestures/switch2/.venv/bin/python -m pip install -r vision/gestures/switch2/requirements.txt
```

Download Google's Pose Landmarker Lite model:

```bash
mkdir -p vision/gestures/switch2/models
curl --fail --location 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' --output vision/gestures/switch2/models/pose_landmarker_lite.task
```

The model and local environment are ignored by Git. The model URL says
latest, so the download may change later. The copy tested here has SHA256:

```text
59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a
```

## Run it

Check the model without opening the camera:

```bash
vision/gestures/switch2/.venv/bin/python vision/gestures/switch2/demo.py --check
```

Start the demo:

```bash
vision/gestures/switch2/.venv/bin/python vision/gestures/switch2/demo.py --camera 2
```

Camera 2 worked on my Mac on November 14. Your camera number may be
different. Change the number after `--camera` if it opens the wrong one.
On Mac, VS Code also needs camera access in System Settings > Privacy &
Security > Camera. Press q while the preview is selected to stop it.

In VS Code, select `vision/gestures/switch2/.venv/bin/python` with
Python: Select Interpreter. Then use Run Python File for `demo.py`.
The preview is not mirrored. LEFT and RIGHT mean the person's body sides.

## What the detector checks

It uses MediaPipe Pose points 11, 13, and 15: left shoulder, elbow, and
wrist. Both visibility and presence must be at least 0.5, and all three
points must be inside the image.

For now the pose rule uses image coordinates. Image y increases downward:

- Arm down: wrist y is more than 0.10 below shoulder y.
- Arm raised: wrist y is more than 0.08 above shoulder y, and elbow y is
  less than 0.05 below shoulder y.

Those values are fractions of the image height. They are starting values
from this draft, not angles. They may need adjusting for the mounted camera.

A brief confidence dip pauses the timers. Uncertain time does not count
toward the hold. If it lasts 0.3 seconds or longer, the arm has to come
down again. A missing pose or disabled attention resets it immediately.

`Switch2Detector.update(points, now, attention_allowed)` takes the current
pose landmarks, monotonic time in seconds, and an attention flag. It returns
True for one frame when the gesture finishes. The demo handles the light
state separately.

## Still to connect and check

- Havish's real head-pose gate. The demo currently sets
  `test_attention_allowed = True` and labels that on screen.
- The shared controller and the actual switch command format.
- The final elevated front-left camera position.
- Ryan's actual gesture, so we can check there is no overlap.
- Lahari's independent testing and any fixes from that.

The shared root `.venv` had MediaPipe 1.0.1, which crashed during pose
initialization on this Mac. This folder uses a separate environment with
MediaPipe 0.10.21, OpenCV contrib 4.11.0.86, and NumPy 1.26.4. The shared
environment was left alone. Other operating systems still need testing.

The old hand gesture practice is separate. This is the left-arm draft for
the team project. Nothing has been committed or pushed yet.
