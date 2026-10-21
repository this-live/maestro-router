from decimal import Decimal


def to_cents(amount):
    return int((Decimal(str(amount)) * 100).to_integral_value())
