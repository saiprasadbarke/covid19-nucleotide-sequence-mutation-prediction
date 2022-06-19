from json import load
from settings.constants import SAVED_PLOTS_PATH, SAVED_STATS_PATH
from matplotlib import pyplot as plt


def plot_mutations_comparision_graphs():
    path_wuhan_gt = f"{SAVED_STATS_PATH}/sorted_difference_indices_wuhan_gt.json"
    path_wuhan_pred = f"{SAVED_STATS_PATH}/sorted_difference_indices_wuhan_pred.json"
    graph_path = f"{SAVED_PLOTS_PATH}/groundtruth_predicted_comparision.png"
    wuhan_gt_data = load(open(path_wuhan_gt))
    wuhan_pred_data = load(open(path_wuhan_pred))

    x_gt = list(wuhan_gt_data.keys())
    y_gt = list(wuhan_gt_data.values())
    x_pred = list(wuhan_pred_data.values())
    y_pred = list(wuhan_pred_data.values())

    plt.figure(figsize=(60, 20))
    plt.bar(x_gt, y_gt, color="red")
    plt.bar(x_pred, y_pred, color="blue", alpha=0.5)
    plt.xlabel("Site Index")
    plt.xticks(rotation=90)
    plt.ylabel("Frequency of mutations at site")
    plt.title("Overlap graph of ground truth and predicted mutations")
    plt.savefig(graph_path)
    plt.show()
