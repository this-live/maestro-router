# semverlite

Semantic Versioning 2.0.0 helpers for our release tooling (stdlib only).

    python3 -m semverlite compare 1.2.3 1.10.0          # -1
    python3 -m semverlite resolve "^1.2" 1.2.0 1.9.3 2.0.0   # 1.9.3
