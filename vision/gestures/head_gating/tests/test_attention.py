import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from vision.gestures.head_gating.head_pose import (CameraModel, MODEL_POINTS, FORWARD, angle_between,
    estimate_from_points, pose_from_transform, AttentionGate, FACING, NOT_FACING, UNKNOWN)
from vision.gestures.head_gating.evaluate import score


def aimed_rotation(t):
    # Construct a head frame whose -z axis points from head to lens.
    z = np.asarray(t, dtype=float) / np.linalg.norm(t)
    x = np.cross([0.,1.,0.], z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    return np.column_stack((x,y,z))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(42)
        self.camera = CameraModel(np.array([[900.,0,640],[0,900.,360],[0,0,1.]]), np.array([.05,-.02,.001,0.,0.]),1280,720,True)

    def recover(self, rotation, translation):
        rvec, _ = cv2.Rodrigues(rotation)
        points, _ = cv2.projectPoints(MODEL_POINTS,rvec,np.array(translation,dtype=float),self.camera.matrix,self.camera.distortion)
        points = points.reshape(-1,2)
        return estimate_from_points(points,self.camera,np.ptp(points[:,0]))

    def test_canonical_axes(self):
        from vision.gestures.head_gating.head_pose import LANDMARK_IDS
        self.assertLess(MODEL_POINTS[LANDMARK_IDS.index(33),0], MODEL_POINTS[LANDMARK_IDS.index(263),0])
        self.assertLess(MODEL_POINTS[LANDMARK_IDS.index(1),2], 0)
        self.assertGreater(MODEL_POINTS[LANDMARK_IDS.index(152),1], 0)

    def test_center_and_room_positions(self):
        for t in ([0,0,100],[-60,20,150],[60,-40,150],[0,100,150],[0,0,60],[0,0,400]):
            with self.subTest(t=t):
                r = aimed_rotation(t)
                pose = self.recover(r,t)
                self.assertIsNotNone(pose)
                self.assertLess(pose.angular_error,0.1)
                self.assertLess(angle_between(pose.forward,r@FORWARD),0.1)

    def test_zero_rotation_is_not_always_facing(self):
        pose = self.recover(np.eye(3),[70,0,100])
        self.assertGreater(pose.angular_error,30)
        self.assertAlmostEqual(pose.yaw,0,places=2)

    def test_all_four_turns(self):
        for axis in ([1,0,0],[0,1,0]):
            for degrees in (-30,30):
                r,_ = cv2.Rodrigues(np.array(axis,dtype=float)*np.radians(degrees))
                pose = self.recover(r,[0,0,100])
                self.assertAlmostEqual(pose.angular_error,30,places=1)

    def test_camera_frame_rotation_invariance(self):
        q,_ = cv2.Rodrigues(np.array([.4,.2,0.]))
        t=np.array([30.,40.,150.]); r=aimed_rotation(t)
        self.assertAlmostEqual(pose_from_transform(q@r,q@t).angular_error,0,places=5)

    def test_corrupt_and_nonfinite_landmarks_rejected(self):
        rvec=np.zeros(3); t=np.array([0.,0.,100.])
        pts,_=cv2.projectPoints(MODEL_POINTS,rvec,t,self.camera.matrix,self.camera.distortion)
        pts=pts.reshape(-1,2); pts[:6]+=100
        self.assertIsNone(estimate_from_points(pts,self.camera,100))
        pts[0,0]=np.nan
        self.assertIsNone(estimate_from_points(pts,self.camera,100))

    def test_calibration_resize_and_aspect(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'camera.json'
            p.write_text(json.dumps({'image_size':[1280,720],'camera_matrix':self.camera.matrix.tolist(),'distortion':self.camera.distortion.tolist()}))
            resized=CameraModel.load(p,640,360)
            self.assertEqual(resized.matrix[0,0],450)
            with self.assertRaises(ValueError): CameraModel.load(p,640,480)

    def test_mirror_display_does_not_change_estimator(self):
        # The display copy may be mirrored without changing input correspondences.
        image=np.arange(12,dtype=np.uint8).reshape(3,4)
        copy=cv2.flip(image,1)
        np.testing.assert_array_equal(image[:,::-1],copy)
        self.assertLess(self.recover(aimed_rotation([40,0,100]),[40,0,100]).angular_error,.1)


class CalibrationToolTests(unittest.TestCase):
    def test_checkerboard_images_recover_known_camera(self):
        k = np.array([[900.,0,640],[0,900.,360],[0,0,1.]])
        board = np.zeros((700,1000),np.uint8)
        for y in range(7):
            for x in range(10):
                board[y*100:(y+1)*100,x*100:(x+1)*100] = 255 if (x+y)%2 else 0
        source = np.float32([[0,0],[1000,0],[1000,700],[0,700]])
        outer = np.array([[-25.,-25.,0.],[225.,-25.,0.],[225.,150.,0.],[-25.,150.,0.]])
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for i in range(16):
                r = rng.uniform(-.4,.4,3); r[2] *= .4
                t = np.array([rng.uniform(-200,-20),rng.uniform(-130,20),rng.uniform(650,900)])
                dest,_ = cv2.projectPoints(outer,r,t,k,np.zeros(5))
                transform = cv2.getPerspectiveTransform(source,dest.reshape(4,2).astype(np.float32))
                image = cv2.warpPerspective(board,transform,(1280,720),borderValue=180)
                cv2.imwrite(str(folder/f'{i:02d}.png'),image)
            output = folder/'camera.json'
            result = subprocess.run([sys.executable,'vision/gestures/head_gating/calibrate_camera.py',
                '--images',str(folder),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            data = json.loads(output.read_text())
            recovered = np.array(data['camera_matrix'])
            self.assertLess(abs(recovered[0,0]-900),30)
            self.assertLess(abs(recovered[1,1]-900),30)
            self.assertLess(data['rms_pixels'],1.)


class GateTests(unittest.TestCase):
    def test_confirmation_hysteresis_invalid_reset(self):
        g=AttentionGate(consecutive=3,alpha=1)
        pose=lambda a: SimpleNamespace(angular_error=a)
        self.assertEqual(g.update(pose(5)),UNKNOWN)
        self.assertEqual(g.update(pose(5)),UNKNOWN)
        self.assertEqual(g.update(pose(5)),FACING)
        for _ in range(10): self.assertEqual(g.update(pose(16)),FACING)
        g.update(pose(25)); g.update(pose(25))
        self.assertEqual(g.update(pose(25)),NOT_FACING)
        for _ in range(10): self.assertEqual(g.update(pose(16)),NOT_FACING)
        self.assertEqual(g.update(None),UNKNOWN)
        self.assertEqual(g.update(pose(float('nan'))),UNKNOWN)
        self.assertIsNone(g.smoothed)

    def test_single_frame_spike_and_reentry(self):
        g=AttentionGate(consecutive=2,alpha=1)
        p=lambda a:SimpleNamespace(angular_error=a)
        g.update(p(0)); self.assertEqual(g.update(p(0)),FACING)
        self.assertEqual(g.update(p(40)),FACING)
        self.assertEqual(g.update(p(0)),FACING)
        self.assertEqual(g.update(None),UNKNOWN)
        self.assertEqual(g.update(p(0)),UNKNOWN)

    def test_invalid_config(self):
        for kwargs in ({'enter':20,'exit':10},{'consecutive':0},{'alpha':0}):
            with self.assertRaises(ValueError): AttentionGate(**kwargs)

    def test_scoring_unknowns_not_hidden(self):
        rows=[{'label':a,'prediction':b} for a,b in [(FACING,FACING),(FACING,UNKNOWN),(NOT_FACING,FACING),(UNKNOWN,FACING)]]
        result=score(rows)
        self.assertEqual(result['false_positives'],2)
        self.assertEqual(result['false_negatives'],1)
        self.assertEqual(result['recall'],.5)
        self.assertAlmostEqual(result['precision'],1/3)


if __name__ == '__main__': unittest.main()
