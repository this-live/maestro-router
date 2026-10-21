from .errors import EvalError
from .tokenizer import tokenize

# Grammar:
#   expr    := term (("+" | "-") term)*
#   term    := unary (("*" | "/") unary)*
#   unary   := ("-" | "+") unary | power
#   power   := atom ("^" unary)?          (right-associative; binds tighter than a unary minus on its left)
#   atom    := NUMBER | NAME | NAME "(" expr ("," expr)* ")" | "(" expr ")"


def _round(x, ndigits=None):
    return round(x) if ndigits is None else round(x, int(ndigits))


FUNCTIONS = {
    "min": (lambda *a: min(a), 1, None),
    "max": (lambda *a: max(a), 1, None),
    "abs": (abs, 1, 1),
    "round": (_round, 1, 2),
}


class Parser:
    def __init__(self, tokens, env=None):
        self.tokens = tokens
        self.env = env or {}
        self.i = 0

    def peek(self):
        return self.tokens[self.i]

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_op(self, *ops):
        kind, value, _ = self.peek()
        return kind == "OP" and value in ops

    def expect(self, op):
        kind, value, pos = self.take()
        if kind != "OP" or value != op:
            raise EvalError("expected %r at %d" % (op, pos))

    def parse(self):
        value = self.expr()
        kind, v, pos = self.peek()
        if kind != "EOF":
            raise EvalError("unexpected %r at %d" % (v, pos))
        return value

    def expr(self):
        value = self.term()
        while self.at_op("+", "-"):
            op = self.take()[1]
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self):
        value = self.unary()
        while self.at_op("*", "/"):
            op = self.take()[1]
            rhs = self.unary()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise EvalError("division by zero")
                value = value / rhs
        return value

    def unary(self):
        if self.at_op("-", "+"):
            op = self.take()[1]
            value = self.unary()
            return -value if op == "-" else value
        return self.power()

    def power(self):
        base = self.atom()
        if self.at_op("^"):
            self.take()
            exp = self.unary()
            try:
                result = base ** exp
            except (ZeroDivisionError, OverflowError) as e:
                raise EvalError(str(e))
            if isinstance(result, complex):
                raise EvalError("complex result")
            return result
        return base

    def atom(self):
        kind, value, pos = self.take()
        if kind == "NUM":
            return value
        if kind == "NAME":
            if self.at_op("("):
                return self.call(value)
            if value not in self.env:
                raise EvalError("unknown name: %s" % value)
            return self.env[value]
        if kind == "OP" and value == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        raise EvalError("unexpected %s at %d" % ("end of input" if kind == "EOF" else repr(value), pos))

    def call(self, name):
        if name not in FUNCTIONS:
            raise EvalError("unknown function: %s" % name)
        fn, lo, hi = FUNCTIONS[name]
        self.expect("(")
        args = []
        if not self.at_op(")"):
            args.append(self.expr())
            while self.at_op(","):
                self.take()
                args.append(self.expr())
        self.expect(")")
        if len(args) < lo or (hi is not None and len(args) > hi):
            raise EvalError("%s() takes %s arguments, got %d" % (name, lo if lo == hi else "%d+" % lo, len(args)))
        return fn(*args)


def evaluate(text, env=None):
    return Parser(tokenize(text), env).parse()
