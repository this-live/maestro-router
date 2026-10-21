from collections import namedtuple

Coupon = namedtuple("Coupon", "code kind value")  # kind: "percent" (value = %) or "fixed" (value = cents)


def subtotal(cart):
    """cart: list of (sku, qty, unit_cents)."""
    return sum(qty * unit for _, qty, unit in cart)


def apply_coupon(amount, coupon):
    if coupon.kind == "percent":
        return amount - amount * coupon.value // 100
    if coupon.kind == "fixed":
        return max(0, amount - coupon.value)
    raise ValueError("unknown coupon kind %r" % coupon.kind)


def total(cart, coupons=()):
    """Stacking rules (PRICING-212): only the best percent coupon is used and it applies first, then every
    fixed coupon. The total never goes below zero."""
    amount = subtotal(cart)
    percent = [c for c in coupons if c.kind == "percent"]
    if percent:
        amount = apply_coupon(amount, max(percent, key=lambda c: c.value))
    for c in coupons:
        if c.kind != "percent":
            amount = apply_coupon(amount, c)
    return max(0, amount)
