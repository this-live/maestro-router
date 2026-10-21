from .models import Customer, Order, ValidationError, validate_email


def create_order(payload):
    """payload: {"name", "email", "lines": [{"sku", "qty"}]} -> (status, body)"""
    try:
        customer = Customer(payload.get("name", ""), validate_email(payload.get("email")))
        order = Order(customer)
        for line in payload.get("lines") or []:
            order.add(line.get("sku"), line.get("qty"))
    except ValidationError as e:
        return 400, {"error": str(e), "field": e.field}
    if not order.lines:
        return 400, {"error": "lines: empty order", "field": "lines"}
    return 201, order.to_dict()
