import pandas as pd
from pathlib import Path


def read_clade_tabular(file_path: str):
    data = pd.read_table(file_path, header=None)
    id_clade = data.iloc[:, [0, 1]]
    print(id_clade.iloc[1, :])


if __name__ == "__main__":
    clade_filepath = (
        f"{Path.cwd()}/data/countrywise_split/clades_tabular/india_clades.tabular"
    )
    read_clade_tabular(clade_filepath)
