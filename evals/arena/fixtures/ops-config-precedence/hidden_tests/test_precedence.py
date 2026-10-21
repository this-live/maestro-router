import json
import os
import tempfile
import unittest

from app.config import DEFAULTS, ConfigError, load_config


class InDir(unittest.TestCase):
    """Each test runs in an empty temp cwd."""

    def setUp(self):
        self.old = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()

    def write(self, rel, data):
        d = os.path.dirname(rel)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(rel, "w") as f:
            json.dump(data, f)


class Precedence(InDir):
    def test_defaults(self):
        self.assertEqual(load_config([], {}), DEFAULTS)

    def test_layers(self):
        self.write("app.json", {"host": "0.0.0.0", "port": 8080, "workers": 4, "log_level": "warning"})
        cfg = load_config([], {})
        self.assertEqual((cfg["host"], cfg["port"], cfg["workers"], cfg["log_level"]), ("0.0.0.0", 8080, 4, "warning"))
        cfg = load_config([], {"APP_PORT": "9000", "APP_WORKERS": "8"})
        self.assertEqual((cfg["host"], cfg["port"], cfg["workers"]), ("0.0.0.0", 9000, 8))
        cfg = load_config(["--port", "7000"], {"APP_PORT": "9000"})
        self.assertEqual((cfg["port"], cfg["workers"], cfg["log_level"]), (7000, 4, "warning"))
        cfg = load_config(["--log-level", "debug", "--host", "::1"], {"APP_LOG_LEVEL": "error"})
        self.assertEqual((cfg["log_level"], cfg["host"], cfg["port"]), ("debug", "::1", 8080))

    def test_debug_bools(self):
        for raw, want in (("false", False), ("0", False), ("No", False), ("OFF", False), ("TRUE", True),
                          ("1", True), ("yes", True), ("On", True)):
            self.assertIs(load_config([], {"APP_DEBUG": raw})["debug"], want, raw)
        self.assertIs(load_config(["--no-debug"], {"APP_DEBUG": "1"})["debug"], False)
        self.assertIs(load_config(["--debug"], {"APP_DEBUG": "0"})["debug"], True)
        self.write("app.json", {"debug": True})
        self.assertIs(load_config([], {})["debug"], True)
        self.assertIs(load_config([], {"APP_DEBUG": "off"})["debug"], False)

    def test_types(self):
        cfg = load_config([], {"APP_ALLOWED_HOSTS": " a.com, b.com ,c.com", "APP_PORT": "8443"})
        self.assertEqual(cfg["allowed_hosts"], ["a.com", "b.com", "c.com"])
        self.assertIsInstance(cfg["port"], int)
        for var, raw in (("APP_PORT", "eighty"), ("APP_DEBUG", "maybe"), ("APP_WORKERS", "2.5")):
            with self.assertRaises(ConfigError) as cm:
                load_config([], {var: raw})
            self.assertIn(var, str(cm.exception))


class Files(InDir):
    def test_config_flag_relative_to_cwd(self):
        self.write("conf/prod.json", {"port": 1234})
        self.assertEqual(load_config(["--config", "conf/prod.json"], {})["port"], 1234)

    def test_app_config_env(self):
        self.write("conf/stage.json", {"port": 2345})
        self.write("app.json", {"port": 1111})
        self.assertEqual(load_config([], {"APP_CONFIG": "conf/stage.json"})["port"], 2345)
        self.write("conf/other.json", {"port": 3456})
        self.assertEqual(load_config(["--config", "conf/other.json"], {"APP_CONFIG": "conf/stage.json"})["port"], 3456)

    def test_missing_explicit_file(self):
        with self.assertRaises(ConfigError):
            load_config(["--config", "nope.json"], {})
        with self.assertRaises(ConfigError):
            load_config([], {"APP_CONFIG": "nope.json"})
        self.assertEqual(load_config([], {})["port"], 8000)


if __name__ == "__main__":
    unittest.main()
