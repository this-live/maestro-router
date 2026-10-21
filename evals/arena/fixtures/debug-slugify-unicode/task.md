Post URLs are coming out mangled for anything that isn't plain ASCII. A post titled "Café Déjà Vu" got the slug `caf--d-j--vu` and "  Hello,   World!  " became `hello----world`. Long titles can also end up with a trailing `-` after truncation.

Can you fix `slugify` in `blog/slug.py`? Expected behaviour:
- accents are transliterated (`é` -> `e`, `ö` -> `o`), plus `ß` -> `ss`, `æ` -> `ae`, `ø` -> `o`
- any run of other characters becomes a single `-`, no leading/trailing `-`
- `max_length` still applies, and the result never ends with `-`
- characters with no ASCII equivalent (e.g. CJK) are dropped; `unique_slug` already falls back to `post` when the slug is empty

So "Café Déjà Vu" -> `cafe-deja-vu`, "Straße nach Köln" -> `strasse-nach-koln`.
