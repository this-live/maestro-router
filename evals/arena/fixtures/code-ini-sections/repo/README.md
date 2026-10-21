# conf

Minimal config-file parser (we can't use configparser: we need the flat dotted keys and our own error messages).

```python
from conf.parser import load, parse
cfg = load("config/app.ini")   # {"name": "billing", "server.port": "8080", ...}
```
