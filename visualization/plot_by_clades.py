import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def read_clade_tabular(file_path: str) -> dict[str, str]:
    data = pd.read_table(file_path, header=None)
    ids = data.iloc[:, 0]
    clades = data.iloc[:, 1]
    id_clade_dict = {}
    for id, clade in zip(ids, clades):
        id_clade_dict[id] = clade
    return id_clade_dict


def plot_bar_clades(id_clade_dict: dict[str, str], op_path: str):
    clades = list(id_clade_dict.values())
    unique_clades = []
    for clade in clades:
        if clade not in unique_clades:
            unique_clades.append(clade)
    number_of_clades_dict = dict.fromkeys(unique_clades)
    for unique_clade in unique_clades:
        number_of_clades_dict[unique_clade] = clades.count(unique_clade)
    x = [str(clade) for clade in list(number_of_clades_dict.keys())]
    y = list(number_of_clades_dict.values())
    plt.xlabel("Clades")
    plt.title("Frequency of clades")
    plt.figure(figsize=(20, 20))
    plt.xticks(rotation=90)
    plt.bar(x, y)
    plt.savefig(op_path)
    plt.show()


if __name__ == "__main__":
    clade_filepath = (
        f"{Path.cwd()}/data/countrywise_split/clades_tabular/india_clades.tabular"
    )
    clades_histogram_path = f"{Path.cwd()}/plots/clades_india_hist.png"
    id_clade_dict = read_clade_tabular(clade_filepath)
    plot_bar_clades(id_clade_dict=id_clade_dict, op_path=clades_histogram_path)
