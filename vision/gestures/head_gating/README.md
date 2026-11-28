# Head facing the camera lens

This module estimates **head orientation toward the lens**, not eye fixation, attention, or mental focus. A person looking sideways with their eyes while keeping their head aimed at the lens can still be FACING CAMERA. This follows the latest head-orientation specification. The previous iris rejection was removed because uncalibrated pupil ratios do not establish eye contact and caused extra false negatives.

## What was wrong

The previous pipeline used six approximate generic face points with y-up coordinates, extracted Euler angles, and required pitch/yaw near zero. Image coordinates use y-down. A frontal fit could therefore involve a near-180-degree rotation about x, which the zero-angle check rejected. The following were also concrete code defects:

- `solvePnP` translation was discarded: standing off-axis while aiming at the lens was rejected, while a head parallel to the camera axis could be accepted off-axis.
- Input frames were mirrored before fitting a fixed anatomical model. Display mirroring must not change estimation coordinates.
- Focal length was assumed to equal image width, with zero distortion, and no calibration option or warning.
- No reprojection, depth, face-size, clipping, or blur checks; a returned fit was treated as reliable.
- No smoothing or consecutive-frame confirmation. Missing/invalid faces were mixed with confidently not-facing faces.
- The iris check used axis-aligned eyelid bounds and arbitrary ranges without a calibrated gaze model. Widening those ranges did not address head geometry.

These are identified implementation errors, not a claim that every reported failure had the same cause. No labeled footage of those failures was available.

## Correct geometry

`canonical_landmarks.json` contains 16 indexed vertices from Google's MediaPipe v0.10.21 canonical model, in centimeters. Source and license are recorded in the file; `MEDIAPIPE_LICENSE` accompanies the derived data. Eye corners, nose, forehead, chin, cheeks, and mouth corners provide non-coplanar correspondences. Expressions and individual face proportions still introduce error.

The canonical model has +y up and +z out of the face. Multiplication by diag(1,-1,-1) is a proper 180-degree coordinate-frame rotation (determinant +1); it defines an upright frontal head frame with +x right, +y down and forward = (0,0,-1). We center it at the outer-eye midpoint. No fitted-angle offsets are added.

OpenCV solves `X_camera = R X_model + t`. Head-forward is `R @ (0,0,-1)`; the direction from the head origin to the lens is `-t / ||t||`. The angular error is `acos(dot(forward, to_lens))`. A yaw/pitch of zero does not imply zero error away from image center. Display yaw/pitch are derived from the forward vector and do not drive classification. Axes show x red, y green, z blue; the magenta arrow points forward.

The entire comparison is in the camera coordinate system. Camera tilt does not need a manually entered room angle: both vectors rotate together. This does not remove perspective distortion or the need for camera calibration. Input video must be raw/unmirrored; `--mirror` flips the annotated display only.

References:
- https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj
- https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/docs/solutions/face_mesh.md

## Install and run

Use the existing Python 3.11 `.venv` (3.11.14 is suitable; 3.11.9 is not specifically required):

```bash
source .venv/bin/activate
python -m pip install -r vision/gestures/head_gating/requirements.txt
python vision/gestures/head_gating/attention_gate.py
```

Use one person at a time. The default source is webcam 0; choose the mounted webcam with `--camera 1` if needed. Press Q or Escape to quit. Green tint/box means FACING CAMERA, red means NOT FACING, yellow means UNKNOWN. The preview includes fitted landmarks, orientation, yaw/pitch, lens error, FPS and the reason for rejecting a measurement.

```bash
python vision/gestures/head_gating/attention_gate.py --camera 0 --width 1280 --height 720 --calibration camera_calibration.json
```

The actual delivered resolution is used, since cameras can ignore requested dimensions. Without a calibration file, approximate intrinsics are visibly marked. This is for initial experimentation, not evidence of correct room geometry. The code uses the existing pinned MediaPipe FaceMesh API, OpenCV and NumPy; it does not download a gaze model or require a dedicated GPU. On macOS MediaPipe may initialize a graphics context even with CPU inference or `--no-display`.

Camera access must be allowed for the launching application (VS Code/Terminal) in macOS Privacy & Security → Camera. A permission failure is not a detection failure. Stop competing camera applications if capture fails. No video is saved unless `--record` is supplied. `--csv` logs only measurements/status.

## Calibrate the mounted webcam

1. Print a flat checkerboard with **9 by 6 inner corners** (10 by 7 squares), or adjust the CLI dimensions for your board. Measure a square's side in millimeters.
2. Keep the same lens, zoom, focus setting, crop and resolution that will be used in the room. Avoid digital stabilization/cropping. If focus or zoom changes, recalibrate. Camera tilt alone does not change intrinsic calibration.
3. Capture 20–30 sharp views, varying board position across the image (including edges), distance and both tilt directions. Keep the entire board visible. Repeated identical/front-parallel views are insufficient even if the tool accepts them.

```bash
python vision/gestures/head_gating/calibrate_camera.py --capture --images calibration_images --columns 9 --rows 6 --square-mm 25 --width 1280 --height 720 --output camera_calibration.json
```

Press SPACE when corners are found to save each view, Q to finish and fit. At least 12 detected views are required. To refit saved images, omit `--capture`. Inspect RMS and per-view errors in the JSON; blurry or wrongly detected views should be removed and recollected. High errors (e.g. several pixels) warrant investigation, not relaxed detection thresholds. Validate on additional checkerboard views not used for fitting. The tool cannot certify good coverage from view count alone. Strong fisheye lenses require a fisheye-specific model, which this implementation does not provide.

The file includes image size, matrix and distortion. Same-aspect resolution scaling is supported; changed aspect ratio is rejected. Even identical aspect ratio does not prove the camera uses the same sensor crop—recalibrate if the mode changes.

## Tune after mounting

- `--enter 12 --exit 18`: angle in degrees for entering/leaving green. Keep entry below exit. Tune against labeled near-boundary examples, not just one frontal frame.
- `--frames 3 --alpha 0.5`: consecutive valid frames and EMA weight. Larger frame counts/lower alpha smooth more but delay switching. Three frames at 30 FPS take about 0.1 seconds; at low FPS the delay is longer. A missing/invalid frame immediately clears the state to UNKNOWN.
- `--min-face 100`: minimum face width in pixels. Prefer more resolution/closer placement to reducing it; a distant face does not contain enough detail for reliable pose.
- `--min-blur 30`: minimum Laplacian variance after face crop resizing to 128x128. Lighting/camera texture changes this heuristic. Tune using clear vs blurred footage.
- `--max-error 0.03`: maximum reprojection RMS relative to face width, with a 2px floor. At least 80% of model points must be RANSAC inliers; final RMS uses all points. A loose limit can admit occluded or incorrect fits.

No face, clipped face, too-small/blurred face, multiple faces, or a failed pose fit yields UNKNOWN. A large bounding-box jump resets confirmation. This is not identity tracking; one-person clips should not switch subjects without an intervening empty frame. Landmark networks can hallucinate an occluded face with a plausible fit; these checks reduce risk but cannot guarantee detection of every occlusion. Avoid interpreting green as a safety-grade decision.

Room positions beneath the camera or far from it are geometrically supported only when enough face is visible at sufficient resolution. Strong profile, occlusion, back-of-head or poor light should produce UNKNOWN rather than a forced NOT FACING. Head translation/distance depend on the generic face model scale; they are approximate, not measured metric depth. Individual anatomy also shifts the estimated forward direction. Calibration cannot eliminate those monocular/model errors.

## Reproducible footage evaluation

Record raw video and matching predictions explicitly:

```bash
python vision/gestures/head_gating/attention_gate.py --calibration camera_calibration.json --record room_trial.mp4 --csv room_trial.csv
```

For each of left/center/right and near/mid/far room positions, perform 10 cycles: aim head at lens for 2 seconds; turn left ~25°, return; right ~25°, return; look up, return; look down at a phone, return; face a wall; leave the frame. Include below-camera positions where a face remains visible. Repeat under bright, dim and backlit conditions, with glasses if relevant, and with partial face occlusion. Add slow boundary sweeps to assess flicker. Use physical orientation references or a second observer/camera for approximate angles rather than assuming this detector's output is ground truth.

Review **raw** footage independently of predictions. Fill the CSV `label` field per zero-based frame with `FACING CAMERA`, `NOT FACING` or `UNKNOWN`. Use UNKNOWN for invisible/ambiguous faces; leave unlabeled frames blank. Frame indices, rather than the nominal video clock, are the authoritative alignment for live recordings. Document label rules and annotate transitions consistently. Keep tuning clips separate from held-out evaluation clips.

```bash
python vision/gestures/head_gating/evaluate.py room_trial.csv
python vision/gestures/head_gating/attention_gate.py --video room_trial.mp4 --calibration camera_calibration.json --csv rerun.csv --no-display
```

The scorer reports TP, FP, FN, precision, recall, UNKNOWN rate and a three-state confusion matrix. UNKNOWN predictions on true facing frames count as false negatives. FACING predictions on unknown ground truth count as false positives (conservative). Undefined metrics are null; unlabeled frames are skipped. Retain original labels by frame number when scoring reruns, and never copy labels from predictions. Report results per lighting/position/distance as well as overall. Measure switch delay in frames and false transitions per steady hold separately; adjacent frames are correlated and not independent trials.

## Validation performed

```bash
python -m unittest discover -s vision/gestures/head_gating/tests -v
```

Automated geometry tests cover canonical axis signs, synthetic projected faces at room positions/distances, 30-degree turns, camera-frame rotation invariance, off-axis heads with zero yaw, corrupted landmarks, calibration scaling, confirmation/hysteresis/reset, and metric accounting. These validate geometry/code, **not real-person accuracy**.

On this Mac, 13 tests passed, including a calibration-tool integration check that recovers known intrinsics from synthetic checkerboard images. A 15-frame blank-video integration run initialized the actual MediaPipe pipeline and produced only UNKNOWN (verified in CSV). Live capture from the agent process was denied/not authorized, including outside the sandbox; no new head-turn accuracy result is claimed. The prior implementation's camera startup does not validate this implementation. Real camera calibration and labeled room trials remain required.

Requirements use the OpenCV contrib distribution already required by MediaPipe; it supplies the same `cv2` import, so a fresh environment does not need a second `opencv-python` distribution writing the same module files.

`pip check` reports MediaPipe 0.10.21 unsupported on this platform because its installed WHEEL metadata declares x86_64, while its native binding is actually universal x86_64+arm64. The actual arm64 MediaPipe inference integration check passed. This packaging warning is recorded rather than suppressed or hidden by editing package metadata.

## Switch cameras and eye badge

Click the preview window and press a number **0–9** to select that camera index (typically 0 is built-in, 1 is external). The current camera and controls appear on screen. An unavailable camera leaves the existing source running. You can also start directly with `--camera 1`.

Switching resets pose/tracking history. Calibration belongs to a particular camera: a new source uses visibly marked approximate intrinsics; restart with its own `--camera N --calibration path.json` for calibrated operation. Switching is disabled during CSV logging or video recording to preserve a single source per trial.

While FACING CAMERA, the green tint is accompanied by an eye icon and **u see me, i see u**. Both disappear when the facing state ends. The caption stays readable with mirrored display.

## Presentation green hold and chime

The preview now defaults to a more forgiving 16° entry / 22° exit angle. After confirmed green detection, losing detection starts a **5-second countdown** while keeping the green tint and eye badge visible. This includes yellow/UNKNOWN and red/NOT FACING. Confirmed facing during the countdown cancels it; the next loss gets a fresh five seconds. Expiry returns to the detector's current color. Switching cameras clears the hold.

The label `GREEN HOLD: ...s` distinguishes the presentation hold from a current detection. CSV predictions remain the actual detector status for honest evaluation. Video-file countdowns use video time; live countdowns use a monotonic clock. A short macOS Pop chime plays once on entry into green, never repeatedly during the hold or its refresh. Use `--mute` to disable it; audio is disabled in `--no-display` mode. Sound playback is nonblocking.

Use `--hold-seconds 0 --enter 12 --exit 18 --mute` to restore the earlier strict display behavior. The new defaults are tuned for the requested presentation convenience; they have not been validated against new room footage.
