from .parser import load

TRUE = ("1", "true", "yes", "on")


class Settings:
    def __init__(self, name, host, port, debug, db_url, pool_size):
        self.name = name
        self.host = host
        self.port = port
        self.debug = debug
        self.db_url = db_url
        self.pool_size = pool_size

    @classmethod
    def from_file(cls, path):
        cfg = load(path)
        return cls(
            name=cfg.get("name", "app"),
            host=cfg.get("server.host", "127.0.0.1"),
            port=int(cfg.get("server.port", "8000")),
            debug=cfg.get("server.debug", "false").lower() in TRUE,
            db_url=cfg["db.url"],
            pool_size=int(cfg.get("db.pool_size", "5")),
        )
