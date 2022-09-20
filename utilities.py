import datetime
import json
import os


def datetime_string(_dt=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), current=False):
    if current:
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return _dt


def inject_datetime_str(file_path):
    filename, file_extension = os.path.splitext(file_path)
    return f"{filename}_{datetime_string()}{file_extension}"


def json_writer(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def midpoint(p1, p2):
    return tuple((j + i) / 2 for i, j in zip(p1, p2))


def automkdir(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def my_round(x, precision=0, base=5):
    """
    :param x: Number to round
    :param precision: rounded to the multiple of 10 to the power minus precision
    :param base: to the nearest 'base'
    :return: rounded value
    """
    return round(base * round(float(x) / base), int(precision))


def euclidean(p1, p2):
    return pow(sum(pow((a - b), 2) for a, b in zip(p1, p2)), 0.5)
