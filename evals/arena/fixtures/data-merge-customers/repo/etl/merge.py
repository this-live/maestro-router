import sys


def merge_customers(crm_path, shop_path):
    """Merge the two customer exports. See the data team wiki for the rules."""
    raise NotImplementedError


def write_merged(rows, path):
    raise NotImplementedError


if __name__ == "__main__":
    write_merged(merge_customers(sys.argv[1], sys.argv[2]), sys.argv[3])
