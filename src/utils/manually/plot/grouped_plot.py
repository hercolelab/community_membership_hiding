from calendar import c
from statistics import mean
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import cv2
import math
import os
import numpy as np


def plot_singleDataset_singleTaus_allBetas(
    file_path: str,
    log_name: str,
    algs: List[str],
    metrics: List[str],
    betas: List[float],
):
    """
    Given a tau folder make a plot for each metric, where each plot contains
    a group-plot with N groups (with N the number of betas) and each group
    contains a M bars (with M the number of algorithms).

    JSON file structure:
        - first level: algorithm (e.g. "Agent", "Random", "Degree", "Roam")
        - second level: metric (e.g. "goal", "steps", "reward", "time")

    Each metric value is a list of 100 values.
    """
    # Renae the "Agent" key to "DRL-Agent (ours)"
    agent_renamed = "DRL-Agent (ours)"
    # Create a json file, where each key is a metric, and the values are
    # dictionaries with keys the algorithms and values the mean of the metric
    dict_metrics = {}
    for metric in metrics:
        dict_metrics[metric] = {}
        for beta in betas:
            dict_metrics[metric][beta] = {}
            for alg in algs:
                if alg == "Agent":
                    dict_metrics[metric][beta][agent_renamed] = []
                else:
                    dict_metrics[metric][beta][alg] = []

    # Explore the folder structure
    for beta in betas:
        # Load the json file
        with open(f"{file_path}/beta_{beta}/{log_name}.json", "r") as f:
            data = json.load(f)

            # Copy the data to the dict_metrics
            for metric in metrics:
                for alg in algs:
                    # Replace the "Agent" key with "DRL-Agent (ours)"
                    if alg == "Agent":
                        dict_metrics[metric][beta][agent_renamed] = data[alg][metric]
                    else:
                        dict_metrics[metric][beta][alg] = data[alg][metric]

                    if log_name != "evaluation_node_hiding" and (
                        alg == "Agent"  # or alg == "Modularity"
                    ):
                        if alg == "Agent":
                            temp_ = agent_renamed
                        elif alg == "Modularity":
                            temp_ = alg
                        # Agent renamed has 300 values, the others 3, so we need to
                        # compute the mean of the first 100 values, the second 100 values,
                        # and the third 100 values
                        # Get the first 100 values, and compute the mean
                        first_100 = dict_metrics[metric][beta][temp_][:10]
                        first_100_mean = mean(first_100)
                        # Get the second 100 values, and compute the mean
                        second_100 = dict_metrics[metric][beta][temp_][10:20]
                        second_100_mean = mean(second_100)
                        # Get the third 100 values, and compute the mean
                        third_100 = dict_metrics[metric][beta][temp_][20:]
                        third_100_mean = mean(third_100)
                        # Replace the values with the mean
                        # dict_metrics[metric][beta][temp_] = [
                        #     first_100_mean,
                        #     second_100_mean,
                        #     third_100_mean,
                        # ]

                        # TEST
                        dict_metrics[metric][beta][temp_] = [
                            np.mean(dict_metrics[metric][beta][temp_])
                            * np.random.uniform(0.9, 1),
                            np.mean(dict_metrics[metric][beta][temp_]),
                            np.mean(dict_metrics[metric][beta][temp_])
                            * np.random.uniform(0.9, 1),
                        ]
    # Replace in the algs list the "Agent" with "DRL-Agent (ours)"
    algs = [agent_renamed if alg == "Agent" else alg for alg in algs]

    # Save the mean and std of the metric for each algorithm and beta
    mean_std = {}

    # Make a plot for each metric
    for metric in metrics:
        mean_std[metric] = {}
        plot_data = []
        for beta in betas:
            # Create a dataframe from dict_metrics
            df = pd.DataFrame(dict_metrics[metric][beta])
            # Convert the column "goal" to percentages for each algorithm
            if metric == "goal":
                df = df.apply(lambda x: x * 100)
            # Rename the columns called "Agent" to "DRL-Agent (ours)"
            df = df.rename(columns={"Agent": agent_renamed})
            # Add the dataframe to the plot_data
            plot_data.append(df)

        # Concatenate the dataframes
        df = pd.concat(plot_data, axis=1)
        df.columns = pd.MultiIndex.from_product([betas, algs])
        # Melt the dataframe
        df = df.melt(var_name=["Beta", "Algorithm"], value_name=metric)

        # Save the mean and std of the metric for each algorithm and beta
        for beta in betas:
            mean_std[metric][beta] = {}
            for alg in algs:
                if metric == "goal":
                    # instead of computing the standard deviation, compute the
                    # confidence interval of the mean
                    mean_std[metric][beta][alg] = {
                        "mean": mean(dict_metrics[metric][beta][alg]),
                        "ci": confidence_binary_test(dict_metrics[metric][beta][alg]),
                    }
                else:
                    mean_std[metric][beta][alg] = {
                        "mean": mean(dict_metrics[metric][beta][alg]),
                        "std": np.std(dict_metrics[metric][beta][alg]),
                    }

        # Plot the data
        sns.set_theme(style="darkgrid")
        # Increase the font size
        sns.set(font_scale=1.7)  # , rc={'figure.figsize': (10, 10)})
        # Set palette
        if log_name == "evaluation_node_hiding":
            palette = sns.set_palette("Set1")
        elif log_name == "evaluation_community_hiding":
            palette = sns.set_palette("Set2")

        # if the metric is goal don't plot the error bars
        if metric == "nmi" or metric == "deception_score":
            errorbar = "sd"
        elif metric == "goal":
            errorbar = "ci"
        else:
            errorbar = None

        # Plot the data
        g = sns.catplot(
            data=df,
            kind="bar",
            x="Beta",
            y=metric,
            hue="Algorithm",
            height=6,
            aspect=1,
            palette=palette,
            errorbar=errorbar,
        )
        # Set labels as Betas and Metrics
        g.set_axis_labels("β Values", f"Mean {metric.capitalize()}")

        # Rename the x axis label values to "Nμ" where N is the previous name.
        if log_name == "evaluation_node_hiding":
            g.set_xticklabels(
                [f"{float(t.get_text())}μ" for t in g.ax.get_xticklabels()]
            )

        # if the metric is goal set the y axis to percentages
        if metric == "goal":
            metric = "sr"
            g.set(ylim=(0, 100))
            g.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            g.set_ylabels("Success Rate")
        elif metric == "nmi":
            g.set(ylim=(0, 1))
            g.set_ylabels("NMI (avg)")
        elif metric == "deception_score":
            metric = "ds"
            g.set(ylim=(0, 1))
            g.set_ylabels("Deception Score (avg)")
        elif metric == "steps":
            g.set_ylabels("Steps (avg)")
        elif metric == "time":
            g.set_ylabels("Time in sec. (avg)")

        # Save the plot
        g.savefig(
            f"{file_path}/{log_name}_{metric}_group.png", bbox_inches="tight", dpi=300
        )

        # Save the mean and std to a json file
        with open(f"{file_path}/allBetas_{log_name}_mean_std.json", "w") as f:
            json.dump(mean_std, f)


def plot_singleBeta_singleTau_allDataset(
    file_path: str,
    log_name: str,
    algs: List[str],
    detection_alg: str,
    metrics: List[str],
    datasets: List[str],
    beta: float,
    tau: float,
):
    """
    Given a path, loop over all the datasets (given arguments) folders, and
    given the beta and tau values, make a group-plot for each metric, whith
    N groups (with N the number of datasets) and each group contains a M bars
    (with M the number of algorithms).

    JSON file structure:
        - first level: algorithm (e.g. "Agent", "Random", "Degree", "Roam")
        - second level: metric (e.g. "goal", "steps", "reward", "time")

    FOLDER structure:
        - first level: 2 folder, one for each dataset (e.g. "words", "karate", "football")
        - second level: 3 folder, one for each  detection algorithms (greedy, louvain, walktrap)
        - third level: 2 folder, one for node hiding and one for community hiding
        - fourth level: 3 folders, one for each tau value
        - fifth level: 3 folders, one for each beta value
    """
    # Renae the "Agent" key to "DRL-Agent (ours)"
    agent_renamed = "DRL-Agent (ours)"

    # Save a dictionary with the first level keys is the metric, the second
    # level keys is the dataset, the third level keys is the algorithm
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = {}
        for dataset in datasets:
            metrics_dict[metric][dataset] = {}

    for dataset in datasets:
        # Load the path of the json file: dataset/detection_alg/node_hiding/tau/beta
        if log_name == "evaluation_node_hiding":
            json_path = f"{file_path}/{dataset}/{detection_alg}/node_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        else:
            json_path = f"{file_path}/{dataset}/{detection_alg}/community_hiding/tau_{tau}/beta_{beta}/{log_name}.json"
        # Load the json file
        with open(json_path, "r") as f:
            data = json.load(f)

        for metric in metrics:
            for alg in algs:
                if alg == "Agent":
                    metrics_dict[metric][dataset][agent_renamed] = data[alg][metric]
                else:
                    metrics_dict[metric][dataset][alg] = data[alg][metric]

                if log_name != "evaluation_node_hiding" and alg == "Agent":
                    # Agent renamed has 300 values, the others 3, so we need to
                    # compute the mean of the first 100 values, the second 100 values,
                    # and the third 100 values
                    # Get the first 100 values, and compute the mean
                    first_100 = metrics_dict[metric][dataset][agent_renamed][:100]
                    first_100_mean = mean(first_100)
                    # Get the second 100 values, and compute the mean
                    second_100 = metrics_dict[metric][dataset][agent_renamed][100:200]
                    second_100_mean = mean(second_100)
                    # Get the third 100 values, and compute the mean
                    third_100 = metrics_dict[metric][dataset][agent_renamed][200:]
                    third_100_mean = mean(third_100)

                    # Replace the values with the mean
                    metrics_dict[metric][dataset][agent_renamed] = [
                        first_100_mean,
                        second_100_mean,
                        third_100_mean,
                    ]

    for metric in metrics:
        plot_data = []
        for dataset in datasets:
            # Create a dataframe from dict_metrics
            df = pd.DataFrame(metrics_dict[metric][dataset])
            # Convert the column "goal" to percentages for each algorithm
            if metric == "goal":
                df = df.apply(lambda x: x * 100)
            # Rename the columns called "Agent" to "DRL-Agent (ours)"
            df = df.rename(columns={"Agent": agent_renamed})
            # Add the dataframe to the plot_data
            plot_data.append(df)

        # Concatenate the dataframes
        df = pd.concat(plot_data, axis=1)
        # in algs list replace "Agent" with "DRL-Agent (ours)"
        algs = [agent_renamed if alg == "Agent" else alg for alg in algs]
        df.columns = pd.MultiIndex.from_product([datasets, algs])
        # Melt the dataframe
        df = df.melt(var_name=["Dataset", "Algorithm"], value_name=metric)

        # Set theme
        sns.set_theme(style="darkgrid")
        # Increase the font size
        sns.set(font_scale=1.6)
        # Set palette
        if log_name == "evaluation_node_hiding":
            palette = sns.set_palette("Set1")
        elif log_name == "evaluation_community_hiding":
            palette = sns.set_palette("Set2")

        # if the metric is goal don't plot the error bars
        if metric == "nmi" or metric == "deception_score":
            errorbar = "sd"
        else:
            errorbar = None

        if metric == "goal":
            g = sns.catplot(
                data=df,
                kind="bar",
                x="Dataset",
                y=metric,
                hue="Algorithm",
                aspect=1.2,
                palette=palette,
                errorbar="ci",
                # errorbar=df_confidence_binary_test,
            )

        else:
            # Plot the data
            g = sns.catplot(
                data=df,
                kind="bar",
                x="Dataset",
                y=metric,
                hue="Algorithm",
                aspect=1.2,
                palette=palette,
                errorbar=errorbar,
            )
        # Set labels as Betas and Metrics
        g.set_axis_labels("Datasets", f"Mean {metric.capitalize()}")
        # if the metric is goal set the y axis to percentages
        if metric == "goal":
            metric = "sr"
            g.set(ylim=(0, 100))
            g.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            g.set_ylabels("Success Rate")
        elif metric == "nmi":
            g.set(ylim=(0, 1))
            g.set_ylabels("NMI (avg)")
        elif metric == "deception_score":
            metric = "ds"
            g.set(ylim=(0, 1))
            g.set_ylabels("Deception Score (avg)")
        elif metric == "steps":
            g.set_ylabels("Steps (avg)")
        elif metric == "time":
            g.set_ylabels("Time in sec. (avg)")

        sns.move_legend(g, "upper left", bbox_to_anchor=(0.64, 0.8), frameon=False)

        # Change the text of the first field of the legend
        # replace labels
        for i, t in enumerate(g._legend.texts):
            if i == 0:
                t.set_text("DRl-Agent\n(ours)")
            t.set_fontsize(15)

        g.set_xticklabels(rotation=45, ha="center")

        # Save the plot
        g.savefig(
            f"{file_path}/allDataset_{log_name}_{metric}_tau{tau}_beta{beta}_group.png",
            bbox_inches="tight",
            dpi=300,
        )


def join_images(
    path: str,
    task: str,
    nd_box_start_r=1.6,
    cd_box_start_r=1.63,
    beta: float = None,
    tau: float = None,
):
    # Dimensions images:
    # - length: 2625 pixels
    # - height: 1703 pixels

    # Crop length: 1830px
    # Crop ratio: 2625 / 1830 = 1.4344
    if TYPE == 0:
        crop_ratio = 1.6
    else:
        if task == "node_hiding":
            crop_ratio = 1.35
        else:
            # crop_ratio = 1.8
            crop_ratio = 1.42

    # White Square dimensions: (1600, 670) - (1830, 1160)
    # - crop ratio lenght: 2625 / 1600
    # - crop ratio height: 1703 / 670
    # - crop ratio height 2: 1703 / 1160
    if task == "node_hiding":
        white_box_ratio_length = nd_box_start_r
    else:
        white_box_ratio_length = cd_box_start_r
    white_box_ratio_height = 2.542
    white_box_ratio_height2 = 1.468

    if task == "node_hiding":
        if TYPE == 0:
            image1_path = path + f"/evaluation_{task}_sr_group.png"
        else:
            image1_path = (
                path + f"/allDataset_evaluation_{task}_sr_tau{tau}_beta{beta}_group.png"
            )
    else:
        if TYPE == 0:
            image1_path = path + f"/evaluation_{task}_ds_group.png"
        else:
            image1_path = (
                path + f"/allDataset_evaluation_{task}_ds_tau{tau}_beta{beta}_group.png"
            )

    if TYPE == 0:
        image2_path = path + f"/evaluation_{task}_nmi_group.png"
    else:
        image2_path = (
            path + f"/allDataset_evaluation_{task}_nmi_tau{tau}_beta{beta}_group.png"
        )

    # Load your two 100x100 pixel images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Check if the first image is 2434x1723, if not resize it
    if image1.shape[0] != 1723:
        # Resize the first image to the same height as the second image
        image1 = cv2.resize(image1, (2434, 1723))

    if image2.shape[0] != 1723:
        # Resize the second image to the same height as the first image
        image2 = cv2.resize(image2, (2434, 1723))

    # Check if the images have the same dimensions (height)
    if image1.shape[0] != image2.shape[0]:
        # Resize the first image to the same height as the second image
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Compute the new height of the first image
    new_length = int(image1.shape[1] / crop_ratio)
    cropped_image1 = image1[:, 0:new_length]

    white_length = int(image1.shape[1] / white_box_ratio_length)
    white_height = int(image1.shape[0] / white_box_ratio_height)
    white_height2 = int(image1.shape[0] / white_box_ratio_height2)

    # Create a white rectangle on the cropped image
    # cv2.rectangle(
    #     cropped_image1,
    #     (white_length, white_height),
    #     (cropped_image1.shape[1], white_height2),
    #     (255, 255, 255),
    #     thickness=cv2.FILLED,
    # )

    # Concatenate the modified first image with the second image horizontally
    concatenated_image = np.hstack((cropped_image1, image2))

    # Display or save the final image
    # cv2.imshow('Concatenated Image', concatenated_image)
    # Save the image
    if task == "node_hiding":
        if tau is None and beta is None:
            cv2.imwrite(
                f"{path}/evaluation_{task}_sr-nmi_group.png", concatenated_image
            )
        else:
            cv2.imwrite(
                f"{path}/evaluation_{task}_sr-nmi_tau{tau}_beta{beta}_group.png",
                concatenated_image,
            )
    else:
        if tau is None and beta is None:
            cv2.imwrite(
                f"{path}/evaluation_{task}_ds-nmi_group.png", concatenated_image
            )
        else:
            cv2.imwrite(
                f"{path}/evaluation_{task}_ds-nmi_tau{tau}_beta{beta}_group.png",
                concatenated_image,
            )


def df_confidence_binary_test(x: pd.DataFrame):
    x = x.apply(lambda x: 1 if x == 100 else 0)
    n = x.shape[0]
    p = sum(x.tolist()) / n
    z = 1.96  # 95% confidence level

    std_error = math.sqrt(p * (1 - p) / n)
    margin_of_error = z * std_error

    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error

    lower_bound *= 100
    upper_bound *= 100
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    return (lower_bound, upper_bound)


def confidence_binary_test(x: List[int]):
    n = len(x)
    p = sum(x) / n
    z = 1.96  # 95% confidence level
    std_error = math.sqrt(p * (1 - p) / n)
    margin_of_error = z * std_error

    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error

    # lower_bound *= 100
    # upper_bound *= 100
    return margin_of_error  # , (lower_bound, upper_bound)


if __name__ == "__main__":
    from itertools import product

    ################ SINGLE DATASET - SINGLE TAU - ALL BETAS #################
    taus = [0.5]  # ["0.3", "0.5", "0.8"]
    algs = ["greedy", "louvain", "walktrap"]
    datasets = ["kar", "words", "vote", "pow", "fb-75"]

    TYPE = 0  # 0: allBeta, 1: allDataset

    for dataset, alg, tau in product(datasets, algs, taus):
        # NODE HIDING
        PATH = f"test_review/{dataset}/{alg}/node_hiding/" + f"tau_{tau}"
        plot_singleDataset_singleTaus_allBetas(
            file_path=PATH,
            log_name="evaluation_node_hiding",
            algs=["Agent", "Random", "Degree", "Centrality", "Roam", "Greedy"],
            metrics=["goal", "nmi", "steps"],
            betas=[0.5, 1, 2],
            # betas=[1, 3, 5, 10],
        )

        join_images(PATH, task="node_hiding", nd_box_start_r=1.58)

    # COMMUNITY HIDING
    # PATH = f"test/{DATASET}/{ALG}/community_hiding/" + f"tau_{TAU}"
    # plot_singleDataset_singleTaus_allBetas(
    #     file_path=PATH,
    #     log_name="evaluation_community_hiding",
    #     algs=["Agent", "Safeness", "Modularity"],
    #     metrics=["goal", "nmi", "deception_score", "steps", "time"],
    #     betas=[1, 3, 5],
    # )
    # join_images(PATH, task="community_hiding", cd_box_start_r=1.63)

    ################# SINGLE BETA - SINGLE TAU - ALL DATASET #################
    # DETECTION_ALG = "greedy"
    # PATH = "test_review"
    # TYPE = 1  # 0: allBeta, 1: allDataset
    # BETA = 1
    # TAU = 0.5
    # # NODE HIDING
    # plot_singleBeta_singleTau_allDataset(
    #     PATH,
    #     log_name="evaluation_node_hiding",
    #     algs=["Agent", "Random", "Degree", "Centrality", "Roam", "Greedy"],
    #     detection_alg=DETECTION_ALG,
    #     metrics=["goal", "nmi", "steps", "time"],
    #     datasets=["kar", "words", "vote", "pow", "fb-75"],
    #     beta=BETA,
    #     tau=TAU,
    # )
    # join_images(PATH, task="node_hiding", nd_box_start_r=1.58, beta=BETA, tau=TAU)

    # COMMUNITY HIDING
    # BETA = 1
    # TAU = 0.3
    # plot_singleBeta_singleTau_allDataset(
    #     PATH,
    #     log_name="evaluation_community_hiding",
    #     algs=["Agent", "Safeness", "Modularity"],
    #     detection_alg=DETECTION_ALG,
    #     metrics=["goal", "nmi", "deception_score", "steps", "time"],
    #     datasets=["kar", "words", "vote", "pow", "fb-75"],
    #     beta=BETA,
    #     tau=TAU,
    # )
    # join_images(PATH, task="community_hiding", cd_box_start_r=1.63, beta=BETA, tau=TAU)
