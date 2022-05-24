from json import load, dump
from pathlib import Path
from typing import List


def trim_input_file(file_path: str, output_files: List[str], train_upto: int, val_upto: int):

    with open(file_path) as f:
        data = load(f)
        train_list = []
        val_list = []
        test_list = []
        for i, item in enumerate(data, 1):
            if i <= train_upto:
                train_list.append(item)
            if i > train_upto and i <= val_upto:
                val_list.append(item)
            else:
                test_list.append(item)

        with open(output_files[0], "w") as out:
            dump(train_list, out)
        with open(output_files[1], "w") as out:
            dump(val_list, out)
        with open(output_files[0], "w") as out:
            dump(test_list, out)


if __name__ == "__main__":
    input_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J.json"
    output_file_paths = [
        f"{Path.cwd().parents[0]}/data/encoded/21A_21J/21A_21J_{dataset}.json" for dataset in ["train", "validation", "test"]
    ]
    trim_input_file(input_file_path, output_file_paths, 24000, 27000)
