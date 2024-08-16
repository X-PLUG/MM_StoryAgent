def parse_list(output):
    try:
        pages = eval(output)
        return isinstance(pages, list)
    except Exception:
        return False