The job queue simulation (`python3 -m jobq.sim`) crashes as soon as two jobs with the same priority are enqueued in the same tick:

```
Traceback (most recent call last):
  File "jobq/sim.py", line 28, in <module>
    main()
  File "jobq/sim.py", line 21, in main
    queue.push(Job(name, flaky(fails), priority=prio))
  File "jobq/queue.py", line 14, in push
    heapq.heappush(self._heap, (priority, self.clock() + delay, job))
TypeError: '<' not supported between instances of 'Job' and 'Job'
```

When I hacked around that locally I hit two more problems: while one job is sitting in retry backoff, nothing else gets processed even though other jobs are ready (the worker just spins until `max_steps`), and retried jobs seem to jump ahead of everything else regardless of their priority.

The intended semantics are in the README. Please fix the queue/worker so the simulation runs and the ordering matches that.
