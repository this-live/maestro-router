from .clock import SimClock
from .job import Job
from .queue import JobQueue
from .worker import Worker


def flaky(fails):
    def run(job):
        if job.attempts <= fails:
            raise RuntimeError("transient failure")
    return run


SCENARIO = [("emails", 1, 0), ("thumbnails", 2, 1), ("billing", 0, 2), ("reports", 2, 0), ("search-index", 1, 0)]


def main():
    clock = SimClock()
    queue = JobQueue(clock)
    for name, prio, fails in SCENARIO:
        queue.push(Job(name, flaky(fails), priority=prio))
    worker = Worker(queue, clock)
    for t, name, what in worker.run_until_idle():
        print("t=%5.1f  %-14s %s" % (t, name, what))


if __name__ == "__main__":
    main()
