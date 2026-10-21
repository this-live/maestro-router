import ast
import os
import unittest

from shop import api, models, validators

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAMES = ("ValidationError", "validate_email", "validate_sku", "validate_quantity")


def tree(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return ast.parse(f.read())


class Extracted(unittest.TestCase):
    def test_defined_in_validators(self):
        for name in NAMES:
            obj = getattr(validators, name)
            self.assertEqual(obj.__module__, "shop.validators", name)
            self.assertIs(getattr(models, name), obj, "models must re-export %s" % name)
        self.assertEqual(validators.MAX_QTY, 999)

    def test_models_has_no_definitions(self):
        defined = {n.name for n in tree("shop/models.py").body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(defined & set(NAMES), set())
        self.assertIn("Order", defined)

    def test_api_and_tests_import_validators(self):
        for rel in ("shop/api.py", "tests/test_validation.py"):
            for node in ast.walk(tree(rel)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("models"):
                    imported = {a.name for a in node.names}
                    self.assertEqual(imported & set(NAMES), set(), "%s still imports validation from models" % rel)

    def test_behaviour(self):
        self.assertEqual(validators.validate_email(" Bob@Example.COM "), "bob@example.com")
        with self.assertRaises(validators.ValidationError) as cm:
            validators.validate_sku("abc")
        self.assertEqual((cm.exception.field, str(cm.exception)), ("sku", "sku: expected AAA-0000"))
        status, body = api.create_order({"name": "Bo", "email": "bo@x.io", "lines": [{"sku": "abc-0001", "qty": 600},
                                                                                    {"sku": "ABC-0001", "qty": 600}]})
        self.assertEqual((status, body), (400, {"error": "quantity: at most 999 per SKU", "field": "quantity"}))
        self.assertEqual(api.create_order({"name": "Bo", "email": "bo@x.io", "lines": [{"sku": "XYZ-9999", "qty": "x"}]}),
                         (400, {"error": "quantity: not a number", "field": "quantity"}))
        self.assertEqual(api.create_order({"name": "Bo", "email": "bo@x.io"}),
                         (400, {"error": "lines: empty order", "field": "lines"}))


if __name__ == "__main__":
    unittest.main()
