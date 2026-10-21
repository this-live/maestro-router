import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
SKU_RE = re.compile(r"^[A-Z]{3}-\d{4}$")
MAX_QTY = 999


class ValidationError(ValueError):
    def __init__(self, field, message):
        super().__init__("%s: %s" % (field, message))
        self.field = field


def validate_email(value):
    value = (value or "").strip().lower()
    if not EMAIL_RE.match(value):
        raise ValidationError("email", "not a valid address")
    return value


def validate_sku(value):
    value = (value or "").strip().upper()
    if not SKU_RE.match(value):
        raise ValidationError("sku", "expected AAA-0000")
    return value


def validate_quantity(value):
    try:
        qty = int(value)
    except (TypeError, ValueError):
        raise ValidationError("quantity", "not a number")
    if not 1 <= qty <= MAX_QTY:
        raise ValidationError("quantity", "must be between 1 and %d" % MAX_QTY)
    return qty
