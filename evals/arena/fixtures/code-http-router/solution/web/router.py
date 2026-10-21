import re

from .response import Response

_PARAM = re.compile(r"<(?:(\w+):)?(\w+)>")
_CONVERTERS = {None: (r"[^/]+", str), "int": (r"[0-9]+", int)}


def _compile(path):
    """(regex, converters, is_static) for a route path."""
    parts, convs, pos = [], {}, 0
    for m in _PARAM.finditer(path):
        conv, name = m.group(1), m.group(2)
        if conv not in _CONVERTERS:
            raise ValueError("unknown converter %r in %r" % (conv, path))
        pattern, fn = _CONVERTERS[conv]
        parts.append(re.escape(path[pos:m.start()]))
        parts.append("(?P<%s>%s)" % (name, pattern))
        convs[name] = fn
        pos = m.end()
    parts.append(re.escape(path[pos:]))
    return re.compile("".join(parts) + r"\Z"), convs, not convs


class Router:
    def __init__(self):
        self.routes = []  # (method, path, regex, converters, is_static, handler)

    def route(self, method, path):
        regex, convs, static = _compile(path)

        def deco(handler):
            self.routes.append((method.upper(), path, regex, convs, static, handler))
            return handler

        return deco

    def _matches(self, path):
        out = []
        for method, _, regex, convs, static, handler in self.routes:
            m = regex.match(path)
            if m:
                kwargs = {k: convs[k](v) for k, v in m.groupdict().items()}
                out.append((not static, method, handler, kwargs))
        out.sort(key=lambda t: t[0])  # static routes first; stable otherwise
        return out

    def dispatch(self, method, path):
        method = method.upper()
        matches = self._matches(path)
        if not matches:
            return Response(404, {"error": "not found"})
        for _, m, handler, kwargs in matches:
            if m == method:
                result = handler(**kwargs)
                return result if isinstance(result, Response) else Response(200, result)
        allow = ", ".join(sorted({m for _, m, _, _ in matches}))
        return Response(405, {"error": "method not allowed"}, {"Allow": allow})
