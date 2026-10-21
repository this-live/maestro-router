from .load import iter_events

STEPS = ("page_view", "add_to_cart", "checkout", "purchase")


def funnel_counts(path):
    """Distinct users that reached each step."""
    users = {s: set() for s in STEPS}
    for e in iter_events(path):
        if e.get("event") in users:
            users[e["event"]].add(e["user"])
    return {s: len(users[s]) for s in STEPS}
