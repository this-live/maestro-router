import json


def export_json(orders):
    data = []
    for o in orders:
        data.append({
            "id": o.id,
            "customer": o.customer,
            "status": o.status,
            "total_cents": o.total_cents,
            "items": [i.to_dict() for i in o.items],
        })
    return json.dumps(data, indent=2) + "\n"
