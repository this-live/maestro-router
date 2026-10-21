from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_decimal(value):
    """Exact Decimal from a str / int / Decimal (floats go through str to avoid binary noise)."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        value = repr(value)
    return Decimal(str(value).strip())


def to_money(value):
    """Round half-up to the cent."""
    return to_decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value):
    return str(to_money(value))


def split_payment(total, n):
    """Split `total` into `n` installments (strings) that add up to `total`; earlier ones get the extra cents."""
    if n < 1:
        raise ValueError("n must be >= 1")
    cents = int(to_money(total) * 100)
    base, extra = divmod(cents, n)
    return [fmt(Decimal(base + (1 if i < extra else 0)) / 100) for i in range(n)]
