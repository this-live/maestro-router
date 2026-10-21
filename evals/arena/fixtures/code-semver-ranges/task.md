Implement `satisfies(version, range_text)` and `max_satisfying(versions, range_text)` in `semverlite/ranges.py` (they're stubs right now; `python3 -m semverlite resolve` depends on them). Use the existing `Version` class for parsing/ordering. We want npm-style ranges:

- primitives `1.2.3` / `=1.2.3`, `>`, `>=`, `<`, `<=` with a full version (whitespace allowed after the operator: `>= 1.2.3`)
- `^`: `^1.2.3` = `>=1.2.3 <2.0.0`, `^0.2.3` = `>=0.2.3 <0.3.0`, `^0.0.3` = `>=0.0.3 <0.0.4`
- `~`: `~1.2.3` = `>=1.2.3 <1.3.0`
- `^`/`~` also take partial versions: `^1.2` = `>=1.2.0 <2.0.0`, `^0.2` = `>=0.2.0 <0.3.0`, `~1.2` = `>=1.2.0 <1.3.0`, `~1` = `>=1.0.0 <2.0.0`
- x-ranges: `1.2`, `1.2.x`, `1.2.*` = `>=1.2.0 <1.3.0`; `1`, `1.x` = `>=1.0.0 <2.0.0`; `*`, `x` and the empty string match anything
- hyphen ranges `1.2.3 - 2.3.4` = `>=1.2.3 <=2.3.4` (full versions on both sides)
- space-separated comparators are ANDed, `||` ORs whole sets
- prereleases (npm rule): a version with a prerelease tag only matches if, in the same AND-set, some comparator's version has a prerelease tag and the same major.minor.patch. So `1.2.3-alpha.7` satisfies `>1.2.3-alpha.3` but `3.4.5-alpha.9` does not, and `*` does not match `1.0.0-rc.1`.
- build metadata is ignored; an invalid range or version raises `ValueError`.

`max_satisfying` returns the highest matching version string as given in the input list, or `None`.
