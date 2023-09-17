from operator import itemgetter

ranges = [
    {
        "base": (255, 0, 0),
        "index": 1,
        "add": True
    },
    {
        "base": (255, 255, 0),
        "index": 0,
        "add": False
    },
    {
        "base": (0, 255, 0),
        "index": 2,
        "add": True
    },
    {
        "base": (255, 255, 0),
        "index": 1,
        "add": False
    },
    {
        "base": (0, 0, 255),
        "index": 0,
        "add": False
    },
    {
        "base": (255, 0, 255),
        "index": 2, # NA
        "add": True # NA
    },
]

def colorize(num):
    """
    Takes number from 0-100 and returns a color
    """
    range_index = int(num//20)
    color_shift = (num/20 - range_index)*255
    base, index, add = itemgetter("base", "index", "add")(ranges[range_index])
    base = list(base)
    if not add:
        color_shift = -color_shift
    base[index] = int(base[index] + color_shift)
    return tuple(base[::-1])
    