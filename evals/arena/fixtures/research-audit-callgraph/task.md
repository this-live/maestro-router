For the SOX review I need to know which public functions in the `billing` package write to the audit log (`billing.audit.record`) when they're called, whether directly or through anything they call (including callbacks they hand to other functions).

"Public function" = a module-level function whose name doesn't start with `_`, in any `billing/*.py` module except `billing/audit.py` itself. Name each one by the module it's defined in, e.g. `billing.invoices.void_invoice`.

Put the answer in `AUDIT_WRITERS.txt` at the repo root: one qualified name per line, sorted, nothing else. Please don't modify the code.
