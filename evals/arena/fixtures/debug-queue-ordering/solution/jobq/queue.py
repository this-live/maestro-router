import heapq
import itertools


class JobQueue:
    """Priority queue of jobs with optional delays (see README for the ordering rules).

    Delayed jobs wait in a heap keyed by ready time; once ready they move to a heap keyed by
    (priority, ready time, push order), so a job in backoff never blocks ready jobs and Job objects are
    never compared."""

    def __init__(self, clock):
        self.clock = clock
        self._ready = []
        self._delayed = []
        self._seq = itertools.count()

    def push(self, job, priority=None, delay=0.0):
        if priority is None:
            priority = job.priority
        heapq.heappush(self._delayed, (self.clock() + delay, next(self._seq), priority, job))

    def _promote(self):
        now = self.clock()
        while self._delayed and self._delayed[0][0] <= now:
            ready_at, seq, priority, job = heapq.heappop(self._delayed)
            heapq.heappush(self._ready, (priority, ready_at, seq, job))

    def pop(self):
        """The most urgent ready job, or None if no job is ready yet."""
        self._promote()
        if not self._ready:
            return None
        return heapq.heappop(self._ready)[3]

    def next_ready_at(self):
        times = [e[1] for e in self._ready] + [e[0] for e in self._delayed]
        return min(times, default=None)

    def __len__(self):
        return len(self._ready) + len(self._delayed)
