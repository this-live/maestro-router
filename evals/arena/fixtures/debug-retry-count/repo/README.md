# netutil

Small helpers shared by our internal services.

- `netutil.retry.retry(times=3, delay=0.0, backoff=2.0, exceptions=(Exception,), sleep=time.sleep)`:
  call the wrapped function up to `times` times in total, sleeping `delay`, `delay*backoff`, ... between attempts.
- `netutil.client.fetch_config(transport)`: fetch and parse the service config, retrying connection errors.
