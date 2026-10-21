from .audit import record as _audit
from .invoices import void_invoice
from .notify import notify


def refund(inv, actor):
    inv["refunded"] = inv["paid"]
    return void_invoice(inv, actor)


def partial_refund(inv, cents, actor):
    inv["refunded"] = inv.get("refunded", 0) + cents
    _audit(actor, "refund.partial", id=inv["id"], cents=cents)
    return inv


def preview_refund(inv):
    """What refund() would do, without side effects (refund() itself goes through audit.record)."""
    return {"id": inv["id"], "refundable": inv["paid"] - inv.get("refunded", 0)}


def escalate(inv, actor):
    return notify("finance", inv, on_sent=_audit)


def schedule_refund(inv, actor):
    return notify("finance", inv)
