class Response:
    def __init__(self, status, body=None, headers=None):
        self.status = status
        self.body = body
        self.headers = dict(headers or {})

    def __repr__(self):
        return "Response(%r, %r, %r)" % (self.status, self.body, self.headers)
