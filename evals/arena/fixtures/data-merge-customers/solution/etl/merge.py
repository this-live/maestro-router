import csv
import sys

from .dates import parse_ts

FIELDS = ["email", "name", "phone", "sources"]


def normalize_email(raw):
    email = (raw or "").strip().lower()
    if "@" not in email:
        return None
    local, _, domain = email.partition("@")
    local = local.split("+", 1)[0]
    if not local or not domain:
        return None
    return local + "@" + domain


def _read(path, source, rank):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            email = normalize_email(row.get("email"))
            if email:
                yield {"email": email, "name": (row.get("name") or "").strip(), "phone": (row.get("phone") or "").strip(),
                       "ts": parse_ts(row.get("updated_at")), "source": source, "rank": rank}


def merge_customers(crm_path, shop_path):
    """Merge the two customer exports. See the data team wiki for the rules."""
    groups = {}
    for rec in list(_read(crm_path, "crm", 1)) + list(_read(shop_path, "shop", 0)):
        groups.setdefault(rec["email"], []).append(rec)
    out = []
    for email in sorted(groups):
        recs = sorted(groups[email], key=lambda r: (r["ts"], r["rank"]), reverse=True)  # newest first, crm on ties
        best = recs[0]
        phone = best["phone"] or next((r["phone"] for r in recs if r["phone"]), "")
        out.append({"email": email, "name": best["name"], "phone": phone,
                    "sources": ";".join(sorted({r["source"] for r in recs}))})
    return out


def write_merged(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in FIELDS})


if __name__ == "__main__":
    write_merged(merge_customers(sys.argv[1], sys.argv[2]), sys.argv[3])
