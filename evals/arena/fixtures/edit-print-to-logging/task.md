Swap the `print()` diagnostics in `syncer/` for the `logging` module so we can ship it as a library.

- every module that logs gets `logger = logging.getLogger(__name__)`
- level comes from the old prefix: `DEBUG:` -> debug, `WARNING:` -> warning, `ERROR:` -> error, no prefix -> info. Drop the prefix, keep the rest of the message text exactly the same
- nothing in the library should write to stdout/stderr anymore
- `syncer/cli.py` keeps printing its final summary line to stdout (that's real output, not a diagnostic); add `--verbose` which configures logging at DEBUG, otherwise INFO
