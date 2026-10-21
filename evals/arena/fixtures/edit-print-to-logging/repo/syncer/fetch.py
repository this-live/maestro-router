def fetch_remote(source):
    print("DEBUG: fetching from %s" % source.name)
    items = source.list()
    if not items:
        print("WARNING: %s returned no items" % source.name)
    print("fetched %d items from %s" % (len(items), source.name))
    return items
