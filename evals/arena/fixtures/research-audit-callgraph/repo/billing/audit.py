_LOG = []


def record(actor, action, **details):
    """Append an entry to the audit log."""
    _LOG.append({"actor": actor, "action": action, "details": details})


def recent(n=10):
    """Read-only view of the last `n` entries."""
    return list(_LOG[-n:])
