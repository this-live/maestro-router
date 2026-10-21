`Settings.from_file("config/app.ini")` can't load our new sectioned config: our little parser in `conf/parser.py` only understands flat `key = value` files. Please extend it:

- `[name]` starts a section; keys after it are stored as `name.key` in the flat dict `parse()` returns. Keys before the first header stay top-level. Whitespace inside the brackets is ignored; an empty `[]` is a `ParseError` that mentions the line number like the existing errors.
- inline comments: a `#` or `;` preceded by whitespace ends the value (`port = 8080   # behind nginx` -> `"8080"`). A `#` with no whitespace before it is part of the value (`url = http://x/#top`).
- a value wrapped in double quotes is taken literally, without the quotes and without comment stripping.
- add `section(cfg, name)` returning `{key: value}` for one section (without the prefix).

Don't break flat files, they're still used everywhere.
