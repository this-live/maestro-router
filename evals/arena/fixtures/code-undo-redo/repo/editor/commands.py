def execute(buf, line):
    """Run one command against `buf`; returns the text to print ('' for none).

    i POS TEXT     insert TEXT at POS
    d POS N        delete N characters at POS
    s POS N TEXT   replace N characters at POS with TEXT
    p              print the buffer
    """
    parts = line.split(" ", 3)
    cmd = parts[0]
    try:
        if cmd == "i":
            buf.insert(int(parts[1]), line.split(" ", 2)[2])
        elif cmd == "d":
            buf.delete(int(parts[1]), int(parts[2]))
        elif cmd == "s":
            buf.replace(int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else "")
        elif cmd == "p":
            return buf.text
        else:
            return "unknown command: %s" % cmd
    except (IndexError, ValueError) as e:
        return "error: %s" % e
    return ""
