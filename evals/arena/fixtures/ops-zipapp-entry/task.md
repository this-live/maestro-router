The release build is broken. We ship `mytool` as a single-file zipapp built by `scripts/build.py`:

```
$ python3 scripts/build.py
built dist/mytool.pyz
$ python3 dist/mytool.pyz --version
Traceback (most recent call last):
  ...
AttributeError: module 'mytool.cli' has no attribute 'main'
```

Also `python3 -m mytool --version` from the repo says `No module named mytool.__main__`.

Please fix the packaging so that `python3 dist/mytool.pyz ARGS` behaves exactly like `python3 -m mytool ARGS` (e.g. `--version` prints `mytool 1.4.2`, `hello Ada` prints `hello, Ada`), and the .pyz works when copied anywhere on its own. `VERSION` at the repo root stays the single source of truth for the version number.
