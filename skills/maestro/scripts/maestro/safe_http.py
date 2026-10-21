"""urllib openers that never carry a credential across origins on a redirect.

urllib's default redirect handling copies every request header (except Content-*) to the redirect target, so an API key
in Authorization, x-api-key or api-key would reach whatever host a provider (or anyone between) redirects to. These
openers follow redirects as usual but drop those headers when the scheme, host or port changes.
"""
from __future__ import annotations

import urllib.parse
import urllib.request

CREDENTIAL_HEADERS = frozenset(("authorization", "proxy-authorization", "x-api-key", "api-key"))


def origin(url):
    u = urllib.parse.urlsplit(url)
    scheme = (u.scheme or "").lower()
    return scheme, (u.hostname or "").lower(), u.port or {"https": 443, "http": 80}.get(scheme)


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and origin(new.full_url) != origin(req.full_url):
            for name in [k for k in new.headers if k.lower() in CREDENTIAL_HEADERS]:
                del new.headers[name]
        return new


def opener(direct=False):
    """An opener that follows redirects without sending credentials to another origin; direct=True ignores proxies."""
    handlers = [_Redirects()]
    if direct:
        handlers.insert(0, urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)
