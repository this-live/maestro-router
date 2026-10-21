from .response import Response


class Router:
    def __init__(self):
        self.routes = {}

    def route(self, method, path):
        def deco(handler):
            self.routes[(method.upper(), path)] = handler
            return handler

        return deco

    def dispatch(self, method, path):
        handler = self.routes.get((method.upper(), path))
        if handler is None:
            return Response(404, {"error": "not found"})
        result = handler()
        return result if isinstance(result, Response) else Response(200, result)
