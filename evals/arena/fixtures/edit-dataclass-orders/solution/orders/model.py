from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LineItem:
    sku: str
    qty: int
    price_cents: int

    def to_dict(self):
        return {"sku": self.sku, "qty": self.qty, "price_cents": self.price_cents}


@dataclass
class Order:
    id: str
    customer: str
    items: List[LineItem] = field(default_factory=list)
    status: str = "new"

    @property
    def total_cents(self):
        return sum(i.qty * i.price_cents for i in self.items)

    def to_dict(self):
        return {"id": self.id, "customer": self.customer, "items": [i.to_dict() for i in self.items],
                "status": self.status, "total_cents": self.total_cents}

    @classmethod
    def from_dict(cls, d):
        return cls(d["id"], d["customer"], [LineItem(i["sku"], i["qty"], i["price_cents"]) for i in d["items"]],
                   d.get("status", "new"))
