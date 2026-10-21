# invoicing

Invoice math for the billing export.

Rounding rules (agreed with accounting, do not change):

1. `line total = qty x unit_price`, rounded **half-up** to the cent.
2. `subtotal = sum of line totals`.
3. `discount = subtotal x discount_pct / 100`, rounded half-up to the cent.
4. `tax = (subtotal - discount) x tax_rate`, rounded half-up to the cent.
5. `total = subtotal - discount + tax`.
6. Installments (`split_payment(total, n)`) must add up exactly to `total`; when it does not divide evenly the
   earlier installments get the extra cents (100.00 / 3 -> 33.34, 33.33, 33.33).

All amounts returned by the API are strings with exactly two decimals (`"7.05"`).

    python3 -m invoicing samples/invoice.csv --tax-rate 0.0825
