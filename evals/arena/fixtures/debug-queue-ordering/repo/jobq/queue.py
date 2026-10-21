import heapq


class JobQueue:
    """Priority queue of jobs with optional delays (see README for the ordering rules)."""

    def __init__(self, clock):
        self.clock = clock
        self._heap = []

    def push(self, job, priority=None, delay=0.0):
        if priority is None:
            priority = job.priority
        heapq.heappush(self._heap, (priority, self.clock() + delay, job))

    def pop(self):
        """The most urgent ready job, or None if no job is ready yet."""
        if not self._heap:
            return None
        priority, ready_at, job = self._heap[0]
        if ready_at > self.clock():
            return None
        heapq.heappop(self._heap)
        return job

    def next_ready_at(self):
        return min((ready_at for _, ready_at, _ in self._heap), default=None)

    def __len__(self):
        return len(self._heap)
