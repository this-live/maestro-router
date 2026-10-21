CATEGORIES = ("tools", "garden", "kitchen")

PRODUCTS = [
    {"sku": "SKU-%03d" % i, "name": "Product %d" % i, "category": CATEGORIES[i % 3], "price_cents": 199 + 37 * i}
    for i in range(1, 48)
]
