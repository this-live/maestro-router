# calc

Safe arithmetic evaluator (no `eval`).

    python3 -m calc "1 + 2 * (3 - 1)"     # 5

`calc.evaluate(text)` returns an int or float; errors raise `calc.EvalError`.
