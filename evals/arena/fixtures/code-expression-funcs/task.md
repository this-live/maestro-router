Extend the calculator in `calc/` (tokenizer + recursive-descent parser) so it can handle the formulas from the pricing sheet:

1. unary `-` and `+` (`2 * -3`, `--4`, `-(1+2)`)
2. `^` for exponentiation: right-associative and tighter than unary minus on its left, like Python's `**`, so `2^3^2` is 512, `-2^2` is -4 and `2^-1` is 0.5
3. variables: `evaluate(text, env=None)` where `env` maps names to numbers; names are `[A-Za-z_][A-Za-z0-9_]*`; an unknown name raises `EvalError` whose message contains the name
4. function calls `min(...)` and `max(...)` (1+ args), `abs(x)`, `round(x)` / `round(x, ndigits)`; wrong number of args or an unknown function raises `EvalError` (message contains the function name for unknown ones)
5. the CLI takes optional `name=value` arguments after the expression: `python3 -m calc "x^2 + y" x=3 y=0.5` prints `9.5`

Anything malformed (`1 +`, `max(1,)`, `min 1`, `2 3`, ...) must raise `EvalError`, never another exception. Keep the existing behaviour and output format.
