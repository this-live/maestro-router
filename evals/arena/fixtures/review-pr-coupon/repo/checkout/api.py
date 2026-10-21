from .pricing import Coupon, subtotal, total

COUPONS = {
    "WELCOME10": Coupon("WELCOME10", "percent", 10),
    "SPRING25": Coupon("SPRING25", "percent", 25),
    "FIVEOFF": Coupon("FIVEOFF", "fixed", 500),
    "SHIPFREE": Coupon("SHIPFREE", "fixed", 799),
}


def quote(cart, codes=()):
    """-> {"subtotal": cents, "total": cents} or {"error": ...}"""
    coupons = []
    for code in codes:
        coupon = COUPONS.get(code.upper())
        if coupon is None:
            return {"error": "unknown coupon %s" % code}
        coupons.append(coupon)
    return {"subtotal": subtotal(cart), "total": total(cart, coupons)}
