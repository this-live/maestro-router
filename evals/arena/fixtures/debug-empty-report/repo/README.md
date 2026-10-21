# stats

Tiny CSV summary tool.

    python3 -m stats report data/sample.csv
    python3 -m stats report data/sample.csv --column latency_ms

Output: `count=<n> mean=<mean, 3 dp> min=<min> max=<max>`.

Run the tests with `python3 -m unittest discover -s tests -t .`
