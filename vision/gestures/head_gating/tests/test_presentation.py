import unittest
from unittest.mock import patch
from vision.gestures.head_gating.presentation import GreenHold, DetectionSound

class PresentationTests(unittest.TestCase):
    def test_loss_countdown_recovery_expiry_and_sound_event(self):
        h = GreenHold(5)
        self.assertEqual(h.update(False, 0), (False, 0, False))
        self.assertEqual(h.update(True, 1), (True, 0, True))
        self.assertEqual(h.update(False, 2), (True, 5, False))
        self.assertEqual(h.update(False, 6), (True, 1, False))
        self.assertEqual(h.update(True, 6.5), (True, 0, False))
        self.assertEqual(h.update(False, 7), (True, 5, False))
        self.assertEqual(h.update(False, 12), (False, 0, False))
        self.assertEqual(h.update(True, 13), (True, 0, True))
        h.reset()
        self.assertEqual(h.update(False, 14), (False, 0, False))

    def test_disable_and_invalid_duration(self):
        h = GreenHold(0)
        h.update(True, 0)
        self.assertEqual(h.update(False, 1), (False, 0, False))
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): GreenHold(value)

    @patch('vision.gestures.head_gating.presentation.subprocess.Popen')
    @patch('vision.gestures.head_gating.presentation.Path.exists', return_value=True)
    @patch('vision.gestures.head_gating.presentation.sys.platform', 'darwin')
    def test_sound_nonblocking_and_no_overlap(self, exists, popen):
        sound = DetectionSound()
        sound.play()
        popen.return_value.poll.return_value = None
        sound.play()
        popen.assert_called_once()
        sound.close()
        popen.return_value.terminate.assert_called_once()
        DetectionSound(False).play()
        popen.assert_called_once()
