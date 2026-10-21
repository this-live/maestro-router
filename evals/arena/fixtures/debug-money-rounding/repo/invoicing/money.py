def to_money(value):
    """Round to the cent."""
    return round(float(value), 2)


def fmt(value):
    return "%.2f" % value


def split_payment(total, n):
    """Split `total` into `n` installments (strings) that add up to `total`."""
    if n < 1:
        raise ValueError("n must be >= 1")
    share = to_money(float(total) / n)
    return [fmt(share) for _ in range(n)]
