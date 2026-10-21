import contextlib
import io
import unittest

from jobq import sim
from jobq.clock import SimClock
from jobq.job import Job
from jobq.queue import JobQueue
from jobq.worker import Worker


def noop(job):
    pass


def fails_first(n):
    def run(job):
        if job.attempts <= n:
            raise RuntimeError("boom")
    return run


def drain(q):
    out = []
    while True:
        j = q.pop()
        if j is None:
            return out
        out.append(j.name)


class Queue(unittest.TestCase):
    def test_same_tick_fifo(self):
        q = JobQueue(SimClock())
        for name in "abcde":
            q.push(Job(name, noop, priority=1))
        self.assertEqual(drain(q), list("abcde"))

    def test_many_same_tick(self):
        q = JobQueue(SimClock())
        seed, expect = 7, []
        for i in range(200):
            seed = (seed * 1103515245 + 12345) % (2 ** 31)
            prio = seed % 4
            q.push(Job("j%d" % i, noop, priority=prio))
            expect.append((prio, i))
        self.assertEqual(drain(q), ["j%d" % i for _, i in sorted(expect)])

    def test_delayed_urgent_does_not_block(self):
        clock = SimClock()
        q = JobQueue(clock)
        q.push(Job("urgent", noop, priority=0), delay=10)
        q.push(Job("normal", noop, priority=5))
        self.assertEqual(q.pop().name, "normal")
        self.assertIsNone(q.pop())
        self.assertEqual(q.next_ready_at(), 10)
        clock.advance(10)
        self.assertEqual(q.pop().name, "urgent")
        self.assertEqual(len(q), 0)

    def test_ready_time_then_fifo(self):
        clock = SimClock()
        q = JobQueue(clock)
        q.push(Job("x", noop, priority=1), delay=5)
        q.push(Job("y", noop, priority=1))
        q.push(Job("p", noop, priority=1), delay=3)
        q.push(Job("q", noop, priority=1), delay=3)
        self.assertEqual(drain(q), ["y"])
        clock.advance(6)
        q.push(Job("z", noop, priority=1))
        q.push(Job("w", noop, priority=0))
        self.assertEqual(drain(q), ["w", "p", "q", "x", "z"])

    def test_len_counts_everything(self):
        q = JobQueue(SimClock())
        q.push(Job("a", noop), delay=3)
        q.push(Job("b", noop))
        self.assertEqual(len(q), 2)
        q.pop()
        self.assertEqual(len(q), 1)


class Workers(unittest.TestCase):
    def run_jobs(self, jobs):
        clock = SimClock()
        q = JobQueue(clock)
        for job, delay in jobs:
            q.push(job, delay=delay)
        w = Worker(q, clock)
        w.run_until_idle(max_steps=500)
        self.assertEqual(len(q), 0)
        return w

    def test_backoff_does_not_stall(self):
        w = self.run_jobs([(Job("flaky", fails_first(2), priority=0), 0), (Job("a", noop, 1), 0),
                           (Job("b", noop, 1), 0), (Job("c", noop, 2), 0)])
        self.assertEqual(w.log, [(0.0, "flaky", "retry"), (1.0, "a", "ok"), (2.0, "flaky", "retry"),
                                 (3.0, "b", "ok"), (4.0, "c", "ok"), (6.0, "flaky", "ok")])

    def test_retry_keeps_priority(self):
        w = self.run_jobs([(Job("s", fails_first(1), priority=5), 0), (Job("u", noop, priority=3), 2)])
        self.assertEqual(w.log, [(0.0, "s", "retry"), (2.0, "u", "ok"), (3.0, "s", "ok")])

    def test_dead_letter(self):
        job = Job("doomed", fails_first(99), priority=1, max_attempts=3)
        w = self.run_jobs([(job, 0)])
        self.assertEqual(w.log, [(0.0, "doomed", "retry"), (2.0, "doomed", "retry"), (6.0, "doomed", "dead")])
        self.assertEqual(w.dead, [job])

    def test_sim_runs(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sim.main()
        lines = buf.getvalue().splitlines()
        self.assertEqual(sum(1 for line in lines if line.endswith(" ok")), 5)
        self.assertEqual(lines[0].split()[2:], ["billing", "retry"])
        self.assertEqual(lines[-1].split(), ["t=", "7.0", "thumbnails", "ok"])


if __name__ == "__main__":
    unittest.main()
