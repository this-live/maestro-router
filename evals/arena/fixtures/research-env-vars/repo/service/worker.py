import os
import subprocess
import sys

QUEUE_PREFIX = "QUEUE_"


def queue_name():
    return os.environ.get(QUEUE_PREFIX + "NAME", "default")


def spawn_child(script):
    env = dict(os.environ, PYTHONUNBUFFERED="1", CHILD_ROLE="worker")
    return subprocess.Popen([sys.executable, script], env=env)
