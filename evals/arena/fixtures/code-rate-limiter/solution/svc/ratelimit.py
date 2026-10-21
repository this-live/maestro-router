class TokenBucket:
    """Starts full; refills continuously at `rate` tokens/second up to `capacity`."""

    def __init__(self, capacity, rate, clock):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.clock = clock
        self.tokens = float(capacity)
        self.updated = clock()

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def allow(self):
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def retry_after(self):
        self._refill()
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.rate
