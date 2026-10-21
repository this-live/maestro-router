# jobq

A deterministic, single-threaded simulation of our background job queue (used to reason about retry storms).

Semantics:

- `JobQueue.push(job, priority=None, delay=0.0)`: `priority` defaults to `job.priority`; lower number = more
  urgent. The job becomes *ready* at `clock() + delay`.
- `JobQueue.pop()` returns the most urgent **ready** job, or `None` if no job is ready yet. Among ready jobs:
  lowest priority number first, then earliest ready time, then push order (FIFO).
- `JobQueue.next_ready_at()` is the earliest ready time of any queued job (`None` when empty).
- `Worker.run_until_idle()` pops and runs jobs; each run takes `job_seconds` of simulated time. A failed job is
  retried with its own priority after `backoff ** attempts` seconds, and goes to `worker.dead` after
  `max_attempts` attempts. When nothing is ready the clock jumps to `next_ready_at()`.
  `worker.log` is a list of `(time, job name, "ok" | "retry" | "dead")`.

    python3 -m jobq.sim
