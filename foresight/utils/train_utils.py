def prefix_dict(prefix, d, seprator='/'):
    """
    Add a prefix to dictionary keys.
    """
    return {prefix + seprator + k: v for k, v in d.items()}


def merge_dict(*args):
    """
    Merge multiple dicts and optionally add a prefix to
    the keys.

    :param args: A list. Each element can be a dict or a
                  tuple (prefix, dict)
    :return: merged dictionary
    """
    ret = dict()
    for arg in args:
        if isinstance(arg, dict):
            ret.update(arg)
        else:
            prefix, d = arg
            ret.update(prefix_dict(prefix, d))
    return ret


def merge_loss_dict(
    full_dict,
    new_dict,
):
    """
    Merge two metadata dictionaries.
    """
    for k, v in new_dict.items():
        if k not in full_dict:
            full_dict[k] = v
        else:
            full_dict[k] = v  # override prior value

    return full_dict