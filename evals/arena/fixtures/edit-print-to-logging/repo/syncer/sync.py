from .fetch import fetch_remote


def sync(source, dest):
    items = fetch_remote(source)
    created = updated = failed = 0
    for item in items:
        try:
            if dest.exists(item.key):
                dest.update(item)
                updated += 1
                print("DEBUG: updated %s" % item.key)
            else:
                dest.create(item)
                created += 1
                print("DEBUG: created %s" % item.key)
        except IOError as e:
            failed += 1
            print("ERROR: could not write %s: %s" % (item.key, e))
    print("sync done: %d created, %d updated, %d failed" % (created, updated, failed))
    return {"created": created, "updated": updated, "failed": failed}
