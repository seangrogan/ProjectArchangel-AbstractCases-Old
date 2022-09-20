import json
import os.path
import warnings


def parfile_reader(parfile_name):
    try:
        filename, file_extension = os.path.splitext(parfile_name)
        if file_extension not in {'.json'}:
            parfile_name = f"{parfile_name}.json"
            warnings.warn(f"Assuming filename {parfile_name}")
        with open(parfile_name) as f:
            pars = json.load(f)
    except(FileNotFoundError):
        raise FileNotFoundError(f"Could not read {parfile_name}")
    return pars


if __name__ == '__main__':
    print(parfile_reader("par1"))

