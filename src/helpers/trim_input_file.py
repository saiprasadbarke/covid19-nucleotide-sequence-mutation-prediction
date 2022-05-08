from json import load, dumps
from pathlib import Path


def trim_input_file(file_path: str, output_file: str, number_of_sequence_pairs: int):

    with open(file_path) as f:
        data = load(f)
        with open(output_file, "w") as out:
            for i, item in enumerate(data, 1):
                # print(item)
                out.write(dumps(item))
                if i == number_of_sequence_pairs:
                    break


if __name__ == "__main__":
    input_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J.json"
    output_file_path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
    trim_input_file(input_file_path, output_file_path, 300)
