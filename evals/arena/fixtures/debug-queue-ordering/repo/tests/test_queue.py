import unittest

from jobq.clock import SimClock
from jobq.job import Job
from jobq.queue import JobQueue


def noop(job):
    pass


class QueueTest(unittest.TestCase):
    def test_priority(self):
        clock = SimClock()
        q = JobQueue(clock)
        q.push(Job("low", noop, priority=5))
        clock.advance(1)
        q.push(Job("high", noop, priority=0))
        self.assertEqual(q.pop().name, "high")
        self.assertEqual(q.pop().name, "low")
        self.assertIsNone(q.pop())

    def test_delay(self):
        clock = SimClock()
        q = JobQueue(clock)
        q.push(Job("later", noop), delay=5)
        self.assertIsNone(q.pop())
        self.assertEqual(q.next_ready_at(), 5)
        clock.advance(5)
        self.assertEqual(q.pop().name, "later")


if __name__ == "__main__":
    unittest.main()
