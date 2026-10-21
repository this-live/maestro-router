# todo

    python3 -m todo --file ~/.todo.json add "Buy milk"
    python3 -m todo --file ~/.todo.json list          # pending items
    python3 -m todo --file ~/.todo.json list --all    # including done
    python3 -m todo --file ~/.todo.json done 1

`--file` defaults to `$TODO_FILE`, then `~/.todo.json`.
