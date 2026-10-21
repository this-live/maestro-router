class Worker:
    def __init__(self, queue, clock, backoff=2.0, job_seconds=1.0):
        self.queue = queue
        self.clock = clock
        self.backoff = backoff
        self.job_seconds = job_seconds
        self.log = []
        self.dead = []

    def run_until_idle(self, max_steps=10000):
        steps = 0
        while len(self.queue) and steps < max_steps:
            steps += 1
            job = self.queue.pop()
            if job is None:
                self.clock.advance_to(self.queue.next_ready_at())
                continue
            job.attempts += 1
            try:
                job.fn(job)
            except Exception:
                if job.attempts >= job.max_attempts:
                    self.dead.append(job)
                    self.log.append((self.clock(), job.name, "dead"))
                else:
                    self.log.append((self.clock(), job.name, "retry"))
                    self.queue.push(job, priority=0, delay=self.backoff ** job.attempts)
            else:
                self.log.append((self.clock(), job.name, "ok"))
            self.clock.advance(self.job_seconds)
        return self.log
