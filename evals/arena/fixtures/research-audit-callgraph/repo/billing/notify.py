def notify(team, inv, on_sent=None):
    """Tell a team about an invoice; `on_sent(actor, action, **details)` is called after sending."""
    message = "[%s] invoice %s needs attention" % (team, inv["id"])
    if on_sent is not None:
        on_sent("notifier", "notify.sent", team=team, id=inv["id"])
    return message
