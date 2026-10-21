import sys

from .config import ConfigError, load_config


def describe(cfg):
    return "serving on %s:%d (workers=%d, debug=%s, log=%s)" % (
        cfg["host"], cfg["port"], cfg["workers"], "on" if cfg["debug"] else "off", cfg["log_level"])


def main(argv=None):
    try:
        cfg = load_config(argv)
    except ConfigError as e:
        print("config error: %s" % e, file=sys.stderr)
        return 2
    print(describe(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
