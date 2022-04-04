# Standard
from json import dump
from pathlib import Path

# External
import matplotlib.pyplot as plt
import pandas as pd


def read_clade_tabular(file_path: str) -> dict[str, str]:
    data = pd.read_table(file_path, header=None)
    ids = data.iloc[:, 0]
    clades = data.iloc[:, 1]
    id_clade_dict = {}
    for id, clade in zip(ids, clades):
        id_clade_dict[id] = clade
    return id_clade_dict


def plot_bar_clades(id_clade_dict: dict[str, str], graph_path: str, stats_path: str):
    clades = list(id_clade_dict.values())
    unique_clades = []
    for clade in clades:
        if clade not in unique_clades:
            unique_clades.append(clade)
    number_of_clades_dict = dict.fromkeys(unique_clades)
    for unique_clade in unique_clades:
        number_of_clades_dict[unique_clade] = clades.count(unique_clade)
    number_of_clades_dict = dict(
        sorted(number_of_clades_dict.items(), key=lambda item: item[1])
    )
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
    clade_filepath = f"{Path.cwd()}/data/complete_sequences/complete_clades.tabular"
    plot_name = "clades_all_hist"
    clades_histogram_path = f"{Path.cwd()}/plots/{plot_name}.png"
    id_clade_dict = read_clade_tabular(clade_filepath)
    stats_file = f"{Path.cwd()}/plots/stats/{plot_name}.json"
    plot_bar_clades(
        id_clade_dict=id_clade_dict,
        graph_path=clades_histogram_path,
        stats_path=stats_file,
    )
