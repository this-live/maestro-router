Accounting flagged a few invoices where our numbers are a cent off from theirs:

- 1 x Widget @ 2.675 is printed as `2.67`, they expect `2.68`
- 7 x Bolt @ 0.145 is printed as `1.01`, they expect `1.02`
- a $100.00 balance split over 3 installments shows `33.33` three times, which only adds up to 99.99

Our rules are in `README.md` (half-up to the cent, per line, then discount, then tax). Please fix `invoicing/` so the amounts are exact. Prices and rates come in as strings from the CSV import; amounts going out stay as 2-decimal strings.
