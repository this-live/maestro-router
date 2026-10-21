from .money import fmt, to_decimal, to_money


class Invoice:
    def __init__(self, tax_rate="0", discount_pct="0"):
        self.tax_rate = to_decimal(tax_rate)
        self.discount_pct = to_decimal(discount_pct)
        self.lines = []

    def add_line(self, description, qty, unit_price):
        self.lines.append({"description": description, "qty": int(qty), "unit_price": to_decimal(unit_price)})

    def _line_total(self, line):
        return to_money(line["qty"] * line["unit_price"])

    def line_totals(self):
        return [fmt(self._line_total(line)) for line in self.lines]

    def totals(self):
        subtotal = sum((self._line_total(line) for line in self.lines), to_money(0))
        discount = to_money(subtotal * self.discount_pct / 100)
        tax = to_money((subtotal - discount) * self.tax_rate)
        total = subtotal - discount + tax
        return {"subtotal": fmt(subtotal), "discount": fmt(discount), "tax": fmt(tax), "total": fmt(total)}
