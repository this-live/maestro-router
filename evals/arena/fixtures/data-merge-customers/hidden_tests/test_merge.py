import os
import tempfile
import unittest

from etl.merge import merge_customers, write_merged

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CRM = os.path.join(ROOT, "data", "crm.csv")
SHOP = os.path.join(ROOT, "data", "shop.csv")

EXPECTED_CSV = """email,name,phone,sources
ann@example.com,Ann Lee-Park,555-0101,crm;shop
bob@example.com,Robert Fox,555-0202,crm;shop
carla@example.org,Carla Diaz,555-0303,crm;shop
dan@example.net,Dan Wu,555-0404,crm
eve@example.com,Eve Stone,555-0505,crm;shop
frank@example.com,Frank Ho,555-0606,shop
"""


def write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


class Merge(unittest.TestCase):
    def test_repo_data(self):
        rows = merge_customers(CRM, SHOP)
        self.assertEqual(rows[0], {"email": "ann@example.com", "name": "Ann Lee-Park", "phone": "555-0101",
                                   "sources": "crm;shop"})
        self.assertEqual([r["email"] for r in rows], ["ann@example.com", "bob@example.com", "carla@example.org",
                                                      "dan@example.net", "eve@example.com", "frank@example.com"])
        self.assertEqual(rows[1]["name"], "Robert Fox")
        self.assertEqual(rows[1]["phone"], "555-0202")
        self.assertEqual(rows[2]["name"], "Carla Diaz")
        self.assertEqual(rows[4]["name"], "Eve Stone")

    def test_write(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.csv")
            write_merged(merge_customers(CRM, SHOP), out)
            with open(out, newline="", encoding="utf-8") as f:
                self.assertEqual(f.read(), EXPECTED_CSV)

    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            crm = write(d, "crm.csv", "email,name,phone,updated_at\n"
                                      "  X+a+b@Mail.COM ,Xavier,,2024-01-01\n"
                                      "y@mail.com,Old Y,555-1,2024-01-01T00:00:00Z\n")
            shop = write(d, "shop.csv", "email,name,phone,updated_at\n"
                                        "x@mail.com,X. Newer,,2024-01-02 00:00\n"
                                        "Y+shop@mail.com,New Y,,02/01/2024 00:01\n"
                                        "y@mail.com,Mid Y,555-2,2024-01-01 12:00\n")
            rows = merge_customers(crm, shop)
        self.assertEqual(rows, [
            {"email": "x@mail.com", "name": "X. Newer", "phone": "", "sources": "crm;shop"},
            {"email": "y@mail.com", "name": "New Y", "phone": "555-2", "sources": "crm;shop"},
        ])


if __name__ == "__main__":
    unittest.main()
