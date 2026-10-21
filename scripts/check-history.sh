#!/bin/sh
# Refuse a git history that contains the private repository's root commit or its old platform/ tree.
# The public repository is a fresh one-commit export; this guards against ever pushing the private history to it.
# Usage: scripts/check-history.sh            (checks every local ref; also works as a pre-push hook)
#        MAESTRO_FORBIDDEN_ROOTS="sha ..."   (override the forbidden root commits, for tests)
set -eu
forbidden="${MAESTRO_FORBIDDEN_ROOTS:-4877a7f}"
status=0
for root in $(git rev-list --max-parents=0 --all); do
    for bad in $forbidden; do
        case "$root" in
            "$bad"*) echo "check-history: refusing: root commit $root is the private repository's root ($bad)" >&2; status=1 ;;
        esac
    done
done
if git log --all --format= --name-only | grep -q '^platform/'; then
    echo "check-history: refusing: the history contains files under platform/ (the old private tree)" >&2
    status=1
fi
[ "$status" -eq 0 ] && echo "check-history: ok ($(git rev-list --all --count) commit(s), roots: $(git rev-list --max-parents=0 --all | cut -c1-7 | tr '\n' ' '))"
exit "$status"
