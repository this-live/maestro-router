from . import audit
from .invoices import create_invoice, invoice_total


class Ledger:
    def post(self, entry):
        audit.record("ledger", "ledger.post", **entry)


def monthly_report(invoices):
    return {"count": len(invoices), "total_cents": invoice_total(invoices)}


def rebill(customer, amount):
    return create_invoice(customer, amount)


def post_to_ledger(entry):
    Ledger().post(entry)


def _backfill(entries):
    for e in entries:
        audit.record("backfill", "ledger.post", **e)
