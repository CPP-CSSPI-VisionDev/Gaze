"""Left-arm gesture for switch 2.

Lower the arm, raise and hold for one second, then lower it to reset.
This returns one event. The demo or shared controller handles ON/OFF.
"""

class Switch2Detector:
    """Give one activation, then wait for the arm to come down."""

    def __init__(self):
        self.armed = False
        self.hold_since = None
        self.rest_since = None
        self.status = "LOWER LEFT ARM"
        self.tracking_lost_since = None

    def update(self, points, now, attention_allowed):
        # No attention or no pose means start again with the arm down.
        if not attention_allowed or not points:
            self.tracking_lost_since = None
            self.armed = False
            self.hold_since = self.rest_since = None
            self.status = (
                "ATTENTION OFF" if not attention_allowed else "NO POSE"
            )
            return False

        shoulder, elbow, wrist = (points[i] for i in (11, 13, 15))

        valid = all(
            min(p.visibility or 0.0, p.presence or 0.0) >= 0.5
            and 0 <= p.x < 1 and 0 <= p.y < 1
            for p in (shoulder, elbow, wrist)
        )

        if not valid:
            if self.tracking_lost_since is None:
                self.tracking_lost_since = now

            if now - self.tracking_lost_since >= 0.3:
                self.armed = False
                self.hold_since = self.rest_since = None

            self.status = "LEFT ARM TRACKING UNCERTAIN"
            return False

        if self.tracking_lost_since is not None:
            gap = now - self.tracking_lost_since

            if gap >= 0.3:
                self.armed = False
                self.hold_since = self.rest_since = None
            else:
                # A short tracking dip pauses the timers instead of losing the gesture.
                if self.hold_since is not None:
                    self.hold_since += gap
                if self.rest_since is not None:
                    self.rest_since += gap

            self.tracking_lost_since = None

        # Image y increases downward.
        # These are image-height margins. We still need to check the mounted angle.
        lowered = wrist.y > shoulder.y + 0.10
        raised = (
            wrist.y < shoulder.y - 0.08
            and elbow.y < shoulder.y + 0.05
        )

        # Start or reset with the arm down for 0.3 seconds.
        if not self.armed:
            self.hold_since = None

            if lowered:
                if self.rest_since is None:
                    self.rest_since = now

                if now - self.rest_since >= 0.3:
                    self.armed = True
                    self.rest_since = None
                    self.status = "READY - RAISE LEFT ARM"
                else:
                    self.status = "CONFIRMING ARM DOWN"
            else:
                self.rest_since = None
                self.status = "LOWER LEFT ARM TO RESET"

            return False

        # Interrupting the raised pose restarts the hold timer.
        if not raised:
            self.hold_since = None
            self.status = "READY - RAISE LEFT ARM"
            return False

        if self.hold_since is None:
            self.hold_since = now

        elapsed = now - self.hold_since
        self.status = f"HOLD LEFT ARM: {min(elapsed, 1.0):.1f}/1.0 s"

        if elapsed < 1.0:
            return False

        # Return True once, then require the arm to come down again.
        self.armed = False
        self.hold_since = self.rest_since = None
        self.status = "TRIGGERED - LOWER LEFT ARM"
        return True
