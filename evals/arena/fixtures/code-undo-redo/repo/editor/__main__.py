import sys

from .buffer import Buffer
from .commands import execute


def main():
    buf = Buffer()
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == "q":
            break
        out = execute(buf, line)
        if out:
            print(out)


if __name__ == "__main__":
    main()
