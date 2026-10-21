"""Pre-2022 helpers kept for the data migration."""
from . import audit


def old_void(inv):
    audit.record("legacy", "invoice.void", id=inv["id"])
    inv["status"] = "void"


def migrate(invoices):
    for inv in invoices:
        if inv.get("status") == "cancelled":
            old_void(inv)
    return invoices
