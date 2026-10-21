class Job:
    def __init__(self, name, fn, priority=0, max_attempts=3):
        self.name = name
        self.fn = fn
        self.priority = priority
        self.max_attempts = max_attempts
        self.attempts = 0

    def __repr__(self):
        return "Job(%r, priority=%r, attempts=%r)" % (self.name, self.priority, self.attempts)
