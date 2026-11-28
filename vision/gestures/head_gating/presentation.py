"""Presentation feedback layered over the detector's actual status."""
import math
import subprocess
import sys
from pathlib import Path


class GreenHold:
    def __init__(self, seconds=5.):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Hold duration must be finite and nonnegative.')
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.lost_at = None
        self.active = False

    def update(self, facing, now):
        was_active = self.active
        if facing:
            self.active = True
            self.lost_at = None
        elif self.active:
            if self.lost_at is None:
                self.lost_at = now
            if now - self.lost_at >= self.seconds:
                self.active = False
        remaining = max(0., self.seconds - (now-self.lost_at)) if self.active and self.lost_at is not None else 0.
        return self.active, remaining, self.active and not was_active


class DetectionSound:
    """Nonblocking macOS chime; audio failure never interrupts capture."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.process = None

    def play(self):
        if not self.enabled or (self.process and self.process.poll() is None):
            return
        sound = Path('/System/Library/Sounds/Pop.aiff')
        if sys.platform == 'darwin' and sound.exists():
            try:
                self.process = subprocess.Popen(['/usr/bin/afplay', str(sound)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
