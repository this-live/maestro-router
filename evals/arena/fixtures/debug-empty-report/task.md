`python3 -m stats report data/empty.csv` blows up:

```
Traceback (most recent call last):
  File "/usr/lib/python3.11/runpy.py", line 198, in _run_module_as_main
  ...
  File "stats/report.py", line 8, in summarize
    mean = total / len(values)
           ~~~~~~^~~~~~~~~~~~~
ZeroDivisionError: division by zero
```

A file with no usable values (header only, or every value blank) shouldn't crash. In that case `summarize` should return `{"count": 0, "mean": None, "min": None, "max": None}` and the report should just print `no data` (exit code 0). Don't change the output for normal files.
