class Response:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.body = body
        self.headers = dict(headers or {})

    def __repr__(self):
        return "Response(%r, %r, %r)" % (self.status, self.body, self.headers)


def json_response(status, body, **headers):
    h = {"Content-Type": "application/json"}
    h.update({k.replace("_", "-"): v for k, v in headers.items()})
    return Response(status, body, h)
