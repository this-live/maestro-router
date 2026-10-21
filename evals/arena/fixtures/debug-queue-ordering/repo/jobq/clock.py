class SimClock:
    """Simulated time; call it to read the current time."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        if seconds < 0:
            raise ValueError("time only moves forward")
        self.now += seconds

    def advance_to(self, t):
        if t is not None and t > self.now:
            self.now = float(t)
