Can you review this PR before I merge it? It's already checked out in the working tree. Fix whatever is wrong in it (keep the intended behaviour from the docstring) and tell me what you found.

> **PRICING-212: allow stacking coupons**
> Checkout can now take several coupon codes. Only the best percent coupon counts and it applies first, then all fixed-amount coupons; the total can't go negative.

```diff
diff --git a/checkout/api.py b/checkout/api.py
--- a/checkout/api.py
+++ b/checkout/api.py
@@ -8,11 +8,12 @@
 }
 
 
-def quote(cart, code=None):
+def quote(cart, codes=()):
     """-> {"subtotal": cents, "total": cents} or {"error": ...}"""
-    coupon = None
-    if code:
+    coupons = []
+    for code in codes:
         coupon = COUPONS.get(code.upper())
         if coupon is None:
             return {"error": "unknown coupon %s" % code}
-    return {"subtotal": subtotal(cart), "total": total(cart, coupon)}
+        coupons.append(coupon)
+    return {"subtotal": subtotal(cart), "total": total(cart, coupons)}
diff --git a/checkout/pricing.py b/checkout/pricing.py
--- a/checkout/pricing.py
+++ b/checkout/pricing.py
@@ -12,12 +12,19 @@
     if coupon.kind == "percent":
         return amount - amount * coupon.value // 100
     if coupon.kind == "fixed":
-        return max(0, amount - coupon.value)
+        return amount - coupon.value
     raise ValueError("unknown coupon kind %r" % coupon.kind)
 
 
-def total(cart, coupon=None):
+def total(cart, coupons=()):
+    """Stacking rules (PRICING-212): only the best percent coupon is used and it applies first, then every
+    fixed coupon. The total never goes below zero."""
     amount = subtotal(cart)
-    if coupon is not None:
-        amount = apply_coupon(amount, coupon)
+    percent = [c for c in coupons if c.kind == "percent"]
+    if percent:
+        best = max(percent, key=lambda c: c.value)
+    for c in coupons:
+        if c.kind == "percent" and c is not best:
+            continue
+        amount = apply_coupon(amount, c)
     return amount
```
