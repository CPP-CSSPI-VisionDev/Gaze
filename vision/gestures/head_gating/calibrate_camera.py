"""Capture checkerboard views, then estimate intrinsics from saved images."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, help='Directory of checkerboard PNG/JPG images')
    p.add_argument('--capture', action='store_true', help='Capture first: SPACE saves a detected board; Q finishes')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--columns', type=int, default=9, help='INNER corner columns, not square count')
    p.add_argument('--rows', type=int, default=6, help='INNER corner rows')
    p.add_argument('--square-mm', type=float, default=25.)
    p.add_argument('--output', default='camera_calibration.json')
    a = p.parse_args()
    if min(a.columns, a.rows) < 3 or a.square_mm <= 0:
        p.error('Use >=3 inner corners per dimension and a positive square size.')
    folder = Path(a.images)
    pattern = (a.columns, a.rows)
    if a.capture:
        folder.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(a.camera)
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
            number = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('No camera frame.')
                found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
                preview = frame.copy()
                if found:
                    cv2.drawChessboardCorners(preview, pattern, corners, found)
                cv2.putText(preview, 'SPACE: save board | Q: finish. Vary position/tilt/distance.', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 1)
                cv2.imshow('Calibration capture', preview)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'),27):
                    break
                if key == 32 and found:
                    while (folder / f'board_{number:04d}.png').exists():
                        number += 1
                    if not cv2.imwrite(str(folder / f'board_{number:04d}.png'), frame):
                        raise RuntimeError('Cannot save calibration image.')
                    print('Saved', number, flush=True)
                    number += 1
        finally:
            cap.release(); cv2.destroyAllWindows()
    grid = np.zeros((a.columns*a.rows,3), np.float32)
    grid[:,:2] = np.mgrid[0:a.columns,0:a.rows].T.reshape(-1,2) * a.square_mm
    objects, images, names = [], [], []
    size = None
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in ('.png','.jpg','.jpeg'):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        current = (image.shape[1], image.shape[0])
        if size is not None and size != current:
            raise ValueError('All calibration images must have the same resolution.')
        size = current
        found, corners = cv2.findChessboardCornersSB(image, pattern)
        if found:
            objects.append(grid.copy()); images.append(corners); names.append(path.name)
    if len(images) < 12:
        raise ValueError(f'Only {len(images)} usable views; collect at least 12 varied views (prefer 20-30).')
    rms, k, distortion, rotations, translations = cv2.calibrateCamera(objects, images, size, None, None)
    errors = []
    for obj, img, r, t in zip(objects, images, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, k, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((projected-img)**2, axis=2)))))
    Path(a.output).write_text(json.dumps({'image_size':size, 'camera_matrix':k.tolist(),
        'distortion':distortion.ravel().tolist(), 'rms_pixels':rms,
        'views':dict(zip(names,errors)), 'pattern':pattern, 'square_mm':a.square_mm}, indent=2)+'\n')
    print(f'Saved {a.output}: {len(images)} views, RMS {rms:.3f}px. Inspect per-view errors and validate on held-out views.')


if __name__ == '__main__':
    main()
