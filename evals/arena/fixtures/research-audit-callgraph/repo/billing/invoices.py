import itertools

from . import audit
from .money import to_cents

_ids = itertools.count(1)


def create_invoice(customer, amount):
    inv = {"id": next(_ids), "customer": customer, "cents": to_cents(amount), "paid": 0, "status": "open"}
    audit.record("system", "invoice.create", id=inv["id"])
    return inv


def void_invoice(inv, actor):
    inv["status"] = "void"
    _log_void(inv, actor)
    return inv


def _log_void(inv, actor):
    audit.record(actor, "invoice.void", id=inv["id"])


def record_payment(inv, amount):
    """Book a payment against an invoice."""
    inv["paid"] += to_cents(amount)
    if inv["paid"] >= inv["cents"]:
        inv["status"] = "paid"
    return inv


def invoice_total(invoices):
    return sum(i["cents"] for i in invoices if i["status"] != "void")


def recent_activity():
    return audit.recent(5)
