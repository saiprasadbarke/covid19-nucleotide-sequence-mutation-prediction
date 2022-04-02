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
    clades = id_clade_dict.values()
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation=90)
    plt.hist(clades, rwidth=0.5)
    plt.savefig(clades_histogram_path)
    plt.show()


if __name__ == "__main__":
    clade_filepath = (
        f"{Path.cwd()}/data/countrywise_split/clades_tabular/india_clades.tabular"
    )
    clades_histogram_path = f"{Path.cwd()}/plots/clades_india_hist.png"
    id_clade_dict = read_clade_tabular(clade_filepath)
    plot_bar_clades(id_clade_dict=id_clade_dict, op_path=clades_histogram_path)
