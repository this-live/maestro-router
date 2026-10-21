from .validators import MAX_QTY, ValidationError, validate_email, validate_quantity, validate_sku

__all__ = ["Customer", "Order", "ValidationError", "validate_email", "validate_quantity", "validate_sku"]


class Customer:
    def __init__(self, name, email):
        self.name = name.strip()
        self.email = validate_email(email)


class Order:
    def __init__(self, customer):
        self.customer = customer
        self.lines = {}

    def add(self, sku, qty):
        sku, qty = validate_sku(sku), validate_quantity(qty)
        total = self.lines.get(sku, 0) + qty
        if total > MAX_QTY:
            raise ValidationError("quantity", "at most %d per SKU" % MAX_QTY)
        self.lines[sku] = total

    def to_dict(self):
        return {"customer": self.customer.email, "lines": [{"sku": s, "qty": q} for s, q in sorted(self.lines.items())]}
