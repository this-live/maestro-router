# Notices

Maestro
Copyright 2026 Bryce Murad
Licensed under the Apache License, Version 2.0 (see LICENSE).

## Third-party material

Maestro bundles data and files from these projects. Their licenses apply to that material.

## models.dev

`skills/maestro/scripts/maestro/data/providers.json` and `data/catalog.json.gz` are built from the [models.dev](https://models.dev) catalog (https://github.com/anomalyco/models.dev) by `evals/build_providers.py` and `evals/build_catalog.py`.

```
MIT License

Copyright (c) 2025 models.dev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face model pages and model cards (numbers only)

`data/catalog.json.gz` also holds numbers that are not in models.dev. They come from two hand-checked tables in `evals/`, merged by `evals/build_catalog.py`:

- `evals/hf_params.json`: parameter counts in billions (total, and active per token where the model card or the repository name states it), read from the Hugging Face Hub model pages on 2026-09-30. The build uses them when the Hub API is not reachable; otherwise it asks the Hub API for the `safetensors.total` count. The catalog also records the Hub repository name of each open-weight model (an identifier such as `org/model`).
- `evals/extra_benchmarks.json`: 630 benchmark results (benchmark name, score, metric) for 50 models, as printed in the vendors' own model cards on huggingface.co (606 rows) and in the gpt-oss model card paper, arXiv 2508.10925 (24 rows). Some card rows are one vendor's evaluation of another vendor's model, and the file says so in each row: 95 rows for 7 open-weight models as tabulated in the GLM-4.7, MiniMax-M2.1 and Step-3.5-Flash cards; 75 rows for 4 closed-model ids (Claude Opus 4.5, Gemini 3 Pro, GPT-5.2) as evaluated in the Kimi-K2.5 card; and 38 rows for 2 open-weight Gemma 4 models as evaluated in the Qwen3.6-35B-A3B card.

Model names, release dates, context sizes and prices in the catalog come from models.dev (above), not from these sources.

These are numeric facts and identifiers about third-party models. Maestro copies no model-card text, tables, charts, weights or code, and no third-party text or code from these sources is redistributed. Each model, model card and paper has the licence its publisher chose; Maestro includes none of them, does not relicense them, and nothing in this file grants rights to them.
