from .errors import EvalError
from .tokenizer import tokenize

# Grammar:
#   expr   := term (("+" | "-") term)*
#   term   := factor (("*" | "/") factor)*
#   factor := NUMBER | "(" expr ")"


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
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
        value = self.factor()
        while self.at_op("*", "/"):
            op = self.take()[1]
            rhs = self.factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise EvalError("division by zero")
                value = value / rhs
        return value

    def factor(self):
        kind, value, pos = self.take()
        if kind == "NUM":
            return value
        if kind == "OP" and value == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        raise EvalError("unexpected %s at %d" % ("end of input" if kind == "EOF" else repr(value), pos))


def evaluate(text):
    return Parser(tokenize(text)).parse()
