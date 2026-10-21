Add undo/redo to the editor buffer (`editor/buffer.py`) and wire it into the command language (`editor/commands.py`).

- `Buffer.undo()` reverts the most recent edit step and `Buffer.redo()` re-applies the most recently undone one. Both return True if they did something, False if there was nothing to do. Any new edit clears the redo history.
- An edit step is one `insert`, `delete` or `replace` call (`replace` is a single step, even though it's currently implemented on top of delete+insert).
- Typing coalescing: a one-character insert of a non-whitespace character, made right where the previous step ended when that step was also such a one-character insert, joins that step. So typing `hello world` one key at a time gives three steps: `hello`, ` `, `world`. An undo/redo always ends the current step.
- Keep at most `max_undo` steps (constructor arg, default 100); drop the oldest.
- Commands: `u` undoes and `r` redoes. They print nothing on success, and `nothing to undo` / `nothing to redo` otherwise.
