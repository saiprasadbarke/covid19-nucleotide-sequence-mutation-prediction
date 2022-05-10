from os.path import isdir
from os import makedirs


def check_dir_exists(path: str):
    check_path = isdir(path)
    if not check_path:
        makedirs(path)
        print(f"Created directory at path: {path}")
