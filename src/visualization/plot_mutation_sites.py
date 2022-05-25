from json import load
from pathlib import Path
import matplotlib.pyplot as plt


def plot_mutations(json_path: str, graph_path: str):
    with open(json_path) as json_file:
        mutation_data = load(json_file)
        sorted_mutation_data = dict(sorted(mutation_data.items(), key=lambda x: int(x[0])))

        sorted_integer_keys = [int(x) for x in sorted_mutation_data.keys()]
        minimum_index = min(sorted_integer_keys)
        maximum_index = max(sorted_integer_keys)

        complete_sequence_mutation_data = {}
        for i in range(1, maximum_index + 1):
            if i in sorted_integer_keys:
                complete_sequence_mutation_data[i] = sorted_mutation_data[str(i)]
            else:
                complete_sequence_mutation_data[i] = 0

        x = list(complete_sequence_mutation_data.keys())
        y = list(complete_sequence_mutation_data.values())

        plt.figure(figsize=(60, 20))
        plt.bar(x, y)
        plt.xlabel("Indices")
        plt.xticks(rotation=90)
        plt.title("Frequency of mutations")
        plt.savefig(graph_path)
        plt.show()


if __name__ == "__main__":
    json_file_path = f"{Path.cwd()}/reports/stats/most_mutated_indices.json"
    graph_path = f"{Path.cwd()}/reports/plots/most_mutated_indices.png"
    plot_mutations(json_file_path, graph_path)
