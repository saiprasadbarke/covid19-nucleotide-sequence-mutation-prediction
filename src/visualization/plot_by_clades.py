# Standard
from json import dump
from pathlib import Path

# External
import matplotlib.pyplot as plt
import pandas as pd

LIST_OF_CLADES = [
    "19A",
    "19B",
    "20A",
    "20B",
    "20C",
    "20E",
    "21A",
    "21B",
    "21D",
    "21H",
    "20D",
    "20F",
    "20I",
    "20J",
    "21M",
    "21E",
    "20G",
    "20H",
    "21C",
    "21F",
    "21I",
    "21J",
    "21G",
    "21K",
    "21L",
    "22A",
    "22B",
    "22C",
]


def read_clade_tabular(file_path: str) -> dict[str, str]:
    data = pd.read_table(file_path, header=None)
    ids = data.iloc[:, 0]
    clades = data.iloc[:, 1]
    qc_status = data.iloc[:, 4]
    id_clade_dict = {}
    number_of_good_sequences = 0
    for idx, (id, clade, qc) in enumerate(zip(ids, clades, qc_status), 1):
        if qc == "good":
            id_clade_dict[id] = str(clade).replace('"', "").split(" ")[0]
            number_of_good_sequences += 1
    print(f"Total sequences: {idx}")
    print(f"Number of good sequences: {number_of_good_sequences}")
    return id_clade_dict


def plot_bar_clades(id_clade_dict: dict[str, str], graph_path: str, stats_path: str):
    clades = list(id_clade_dict.values())
    unique_clades = []
    for clade in clades:
        if clade not in unique_clades:
            unique_clades.append(clade)
    number_of_clades_dict = dict.fromkeys(LIST_OF_CLADES)
    for unique_clade in LIST_OF_CLADES:
        number_of_clades_dict[unique_clade] = clades.count(unique_clade)
    number_of_clades_dict = dict(sorted(number_of_clades_dict.items(), key=lambda item: item[1]))
    with open(stats_path, "w") as fp:
        dump(number_of_clades_dict, fp)
    print(number_of_clades_dict)
    x = [
        str(clade) for clade in list(number_of_clades_dict.keys())
    ]  # Need to perform this step as x has some nan values
    y = list(number_of_clades_dict.values())

    plt.figure(figsize=(20, 20))
    plt.bar(x, y)
    plt.xlabel("Clades")
    plt.xticks(rotation=90)
    plt.title("Frequency of clades")
    plt.savefig(graph_path)
    plt.show()


if __name__ == "__main__":
    clade_filepath = f"{Path.cwd().parents[0]}/data/clades.tabular"
    plot_name = "clades_all_hist_after_qc_filter_spike"
    clades_histogram_path = f"{Path.cwd()}/reports/plots/{plot_name}.png"
    id_clade_dict = read_clade_tabular(clade_filepath)
    print("Printing Graphs now...")
    stats_file = f"{Path.cwd()}/reports/stats/{plot_name}.json"
    plot_bar_clades(
        id_clade_dict=id_clade_dict,
        graph_path=clades_histogram_path,
        stats_path=stats_file,
    )
