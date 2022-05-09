from json import load, dump
from pathlib import Path


def trim_input_file(file_path: str, output_file: str, number_of_sequence_pairs: int):

    with open(file_path) as f:
        data = load(f)
        trimmed_list = []
        for i, item in enumerate(data, 1):
            trimmed_list.append(item)
            if i == number_of_sequence_pairs:
                break

        with open(output_file, "w") as out:
            dump(trimmed_list, out)


if __name__ == "__main__":
    input_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J.json"
    output_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
    trim_input_file(input_file_path, output_file_path, 300)
