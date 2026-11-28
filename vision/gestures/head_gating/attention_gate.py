"""Run: python vision/gestures/head_gating/attention_gate.py --help"""
import argparse
import csv
import time
from contextlib import ExitStack
import cv2
import numpy as np
try:
    from .presentation import GreenHold, DetectionSound
    from .head_pose import CameraModel, AttentionGate, estimate_head_pose, LANDMARK_IDS, FACING, NOT_FACING, UNKNOWN
except ImportError:
    from presentation import GreenHold, DetectionSound
    from head_pose import CameraModel, AttentionGate, estimate_head_pose, LANDMARK_IDS, FACING, NOT_FACING, UNKNOWN


def parser():
    p = argparse.ArgumentParser(description='Head-to-lens orientation; one person at a time.')
    p.add_argument('--hold-seconds', type=float, default=2., help='Keep green after detection is lost; 0 disables')
    p.add_argument('--mute', action='store_true', help='Disable the detection chime')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--video', help='Analyze a recorded video instead of the webcam (unmirrored input).')
    p.add_argument('--calibration', help='Camera calibration JSON')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--enter', type=float, default=16.)
    p.add_argument('--exit', type=float, default=22.)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--min-face', type=int, default=100, help='Minimum face width in pixels')
    p.add_argument('--min-blur', type=float, default=30., help='Minimum Laplacian variance on a 128px face crop')
    p.add_argument('--max-error', type=float, default=0.03, help='Maximum reprojection RMS / face width')
    p.add_argument('--mirror', action='store_true', help='Mirror display only, never estimation')
    p.add_argument('--csv', help='Write frame-indexed predictions for manual evaluation')
    p.add_argument('--record', help='Opt-in raw unmirrored MP4 recording')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    return p


def draw_eye_badge(frame):
    """Draw after display mirroring so the caption stays readable."""
    h, w = frame.shape[:2]
    scale = min(1., w / 640, h / 360)
    cx, cy = w // 2, h - int(72 * scale)
    white, green = (240, 255, 240), (80, 255, 140)
    overlay = frame.copy()
    cv2.rectangle(overlay, (max(0, cx-int(190*scale)), cy-int(35*scale)),
                  (min(w-1, cx+int(190*scale)), h-5), (18, 32, 22), -1)
    cv2.addWeighted(overlay, .8, frame, .2, 0, dst=frame)
    cv2.ellipse(frame, (cx, cy), (max(1,int(39*scale)), max(1,int(20*scale))), 0, 0, 360, white, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(1,int(14*scale)), green, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(1,int(7*scale)), (15, 30, 20), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx-int(4*scale), cy-int(5*scale)), max(1,int(3*scale)), white, -1, cv2.LINE_AA)
    text = 'u see me, i see u'
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .65*scale, 2)[0]
    cv2.putText(frame, text, (cx-size[0]//2, cy+int(43*scale)), cv2.FONT_HERSHEY_SIMPLEX, .65*scale, white, 2, cv2.LINE_AA)


def open_camera(index, width, height):
    candidate = cv2.VideoCapture(index)
    candidate.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if candidate.isOpened():
        ok, frame = candidate.read()
        if ok and frame is not None:
            return candidate
    candidate.release()
    return None


def main():
    args = parser().parse_args()
    gate = AttentionGate(args.enter, args.exit, args.frames, args.alpha)
    hold = GreenHold(args.hold_seconds)
    sound = DetectionSound(not args.mute and not args.no_display)
    if min(args.width, args.height, args.min_face) <= 0 or args.min_blur < 0 or args.max_error <= 0:
        raise ValueError('Resolution and quality thresholds must be positive.')
    import mediapipe as mp
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    writer = None
    active_camera = args.camera
    calibration_path = args.calibration
    notice = ''
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open video/camera. Check the source and camera permission.')
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera, previous_box = None, None
        number, fps, previous_time = 0, 0., time.perf_counter()
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if 0 < source_fps < 240 else 30.
        with ExitStack() as stack:
            mesh = stack.enter_context(mp.solutions.face_mesh.FaceMesh(
                max_num_faces=2, refine_landmarks=False,
                min_detection_confidence=0.7, min_tracking_confidence=0.7))
            log = None
            if args.csv:
                log = csv.writer(stack.enter_context(open(args.csv, 'w', newline='')))
                log.writerow(['frame', 'time_seconds', 'prediction', 'reason', 'angular_error', 'yaw', 'pitch', 'reprojection_px', 'label'])
            while True:
                ok, raw = cap.read()
                if not ok:
                    if not args.video or number == 0:
                        raise RuntimeError('Camera/video stopped delivering frames; no current facing estimate is available.')
                    break
                h, w = raw.shape[:2]
                if camera is None:
                    camera = CameraModel.load(calibration_path, w, h)
                    print('Calibrated intrinsics' if camera.calibrated else 'WARNING: approximate intrinsics; use checkerboard calibration before room evaluation.', flush=True)
                    if args.record:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open recording output.')
                if (w, h) != (camera.width, camera.height):
                    raise RuntimeError('Video resolution changed; calibration no longer applies.')
                if writer:
                    writer.write(raw)
                result = mesh.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                faces = result.multi_face_landmarks or []
                pose, box, reason = None, None, 'No reliable face'
                display = raw.copy()
                if len(faces) == 1:
                    lm = faces[0].landmark
                    xy = np.array([(p.x * w, p.y * h) for p in lm])
                    x1, y1 = np.floor(xy.min(axis=0)).astype(int)
                    x2, y2 = np.ceil(xy.max(axis=0)).astype(int)
                    box = (x1, y1, x2, y2)
                    if previous_box is not None:
                        a, b, c, d = previous_box
                        intersection = max(0, min(c, x2)-max(a, x1)) * max(0, min(d, y2)-max(b, y1))
                        union = (c-a)*(d-b) + (x2-x1)*(y2-y1) - intersection
                        if intersection / max(1, union) < 0.25:
                            gate.reset()
                    if x1 < 2 or y1 < 2 or x2 >= w-2 or y2 >= h-2:
                        reason = 'Face clipped by frame edge'
                    elif x2-x1 < args.min_face or y2 <= y1:
                        reason = 'Face too small'
                    else:
                        crop = cv2.cvtColor(raw[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        blur = cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var()
                        if blur < args.min_blur:
                            reason = 'Face blurred / low texture'
                        else:
                            pose = estimate_head_pose(lm, camera, x2-x1, args.max_error)
                            reason = 'Valid pose' if pose is not None else 'Pose fit unreliable'
                    for i in LANDMARK_IDS:
                        cv2.circle(display, tuple(xy[i].astype(int)), 2, (255, 200, 0), -1)
                elif len(faces) > 1:
                    reason = 'Multiple faces - use one person'
                previous_box = box
                status = gate.update(pose)
                hold_time = number/source_fps if args.video else time.monotonic()
                green, remaining, entered = hold.update(status == FACING, hold_time)
                if entered:
                    sound.play()
                color = {FACING: (0, 220, 0), NOT_FACING: (0, 0, 255), UNKNOWN: (0, 220, 255)}[status]
                if green:
                    tint = np.zeros_like(display); tint[:, :, 1] = 180
                    display = cv2.addWeighted(display, 0.88, tint, 0.12, 0)
                if green:
                    color = (0, 220, 0)
                if box:
                    cv2.rectangle(display, box[:2], box[2:], color, 2)
                if pose is not None:
                    rvec, _ = cv2.Rodrigues(pose.rotation)
                    cv2.drawFrameAxes(display, camera.matrix, camera.distortion, rvec, pose.translation, 3.)
                    arrow, _ = cv2.projectPoints(np.array([[0.,0.,0.], [0.,0.,-5.]]), rvec, pose.translation, camera.matrix, camera.distortion)
                    a, b = arrow.reshape(-1, 2).astype(int)
                    cv2.arrowedLine(display, tuple(a), tuple(b), (255, 0, 255), 2)
                if args.mirror:
                    display = cv2.flip(display, 1)
                if green:
                    draw_eye_badge(display)
                now = time.perf_counter()
                instantaneous = 1 / max(now-previous_time, 1e-6)
                fps = instantaneous if number == 0 else 0.2 * instantaneous + 0.8 * fps
                previous_time = now
                values = '--' if pose is None else f'yaw {pose.yaw:+.1f}  pitch {pose.pitch:+.1f}  lens error {pose.angular_error:.1f} deg'
                headline = f'GREEN HOLD: {remaining:.1f}s | {status}' if remaining > 0 else status + ' | ' + reason
                lines = [headline, values,
                         f'{fps:.1f} FPS | {"CALIBRATED" if camera.calibrated else "APPROXIMATE INTRINSICS"} | Q: quit',
                         ('Video file' if args.video else f'Camera {active_camera} | Press 0-9 to switch') + (' | ' + notice if notice else '')]
                cv2.rectangle(display, (0,0), (w,120), (20,20,20), -1)
                for row, line in enumerate(lines):
                    cv2.putText(display, line, (10,25+row*28), cv2.FONT_HERSHEY_SIMPLEX, min(0.6,w/1200), color, 1, cv2.LINE_AA)
                if log:
                    log.writerow([number, number/source_fps, status, reason,
                                  *(['']*4 if pose is None else [pose.angular_error, pose.yaw, pose.pitch, pose.reprojection_error]), ''])
                number += 1
                if not args.no_display:
                    cv2.imshow('Gaze - head facing lens', display)
                    key = cv2.waitKey(1) & 0xff
                    if key in (ord('q'), 27):
                        break
                    if not args.video and ord('0') <= key <= ord('9'):
                        target = key - ord('0')
                        if args.record or args.csv:
                            notice = 'Stop recording/logging before switching'
                        elif target != active_camera:
                            candidate = open_camera(target, args.width, args.height)
                            if candidate is None:
                                notice = f'Camera {target} unavailable; staying on {active_camera}'
                            else:
                                cap.release()
                                cap = candidate
                                active_camera = target
                                calibration_path = args.calibration if target == args.camera else None
                                camera, previous_box = None, None
                                gate.reset()
                                hold.reset()
                                mesh.reset()
                                fps, previous_time = 0., time.perf_counter()
                                notice = 'Switched' if calibration_path else 'Switched - needs its own calibration'

                if args.max_frames and number >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sound.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
