from json import load, dump
from pathlib import Path
import matplotlib.pyplot as plt


def plot_by_clades_unique_sequences(merged_file: str, graph_path: str, stats_path: str):
    clade_seqlen_dict = {}
    data = load(open(merged_file))
    for clade, sequences_list in data.items():
        clade_seqlen_dict[clade] = len(sequences_list)

    with open(stats_path, "w") as fout:
        dump(clade_seqlen_dict, fout)

    x = list(clade_seqlen_dict.keys())
    y = list(clade_seqlen_dict.values())
    plt.figure(figsize=(60, 20))
    plt.bar(x, y)
    plt.xlabel("Clades")
    plt.xticks(rotation=90, fontsize=5)
    plt.ylabel("Number of unique sequences")
    plt.savefig(graph_path)
    plt.show()


if __name__ == "__main__":
    merged_file = f"{Path.cwd()}/data/merged.json"
    graph_path = f"{Path.cwd()}/reports/plot_by_clades_unique_sequences.png"
    stats_path = f"{Path.cwd()}/reports/stats/plot_by_clades_unique_sequences.json"
    plot_by_clades_unique_sequences(merged_file, graph_path, stats_path)
