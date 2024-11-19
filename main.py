from src.utils.utils import HyperParams, editable_HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding

from src.utils.utils import (
    extrapolate_metrics,
    json_to_md_tables
)

import argparse
import math
import time
import json
import numpy as np


def get_args():
    """
    Function for handling command line arguments

    Returns
    -------
    args : argparse.Namespace
    """
    #parser = argparse.ArgumentParser(description="PyTorch A2C")
    # Mode: train or test
    #parser.add_argument("--mode", type=str, default="both", help="train | test | both")
    # Argument parsing
    #return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    datasets = [
        #FilePaths.KAR.value,
        FilePaths.WORDS.value,
        #FilePaths.VOTE.value,
        #FilePaths.NETS.value,
        #FilePaths.POW.value,
        #FilePaths.FB_75.value,
        #FilePaths.ASTR.value,
    ]
    detection_algs = [
        DetectionAlgorithmsNames.GRE.value,
        #DetectionAlgorithmsNames.LOUV.value,
        #DetectionAlgorithmsNames.WALK.value,
    ]

    results = {
        dataset : dict() for dataset in datasets
    }

    n_experiments = 100

    for dataset in datasets:
        editable_HyperParams.GRAPH_NAME = dataset
        # ° --- Environment Setup --- ° #
        env = GraphEnvironment(graph_path=dataset)

        results[dataset] = dict()

        # ° ------  Agent Setup ----- ° #
        agent = Agent(env=env)

        for e in range(n_experiments):
            print(f"******************** Experiment {e+1} ********************")

            # ° ------    TRAIN    ------ ° #
            #if args.mode == "train" or args.mode == "both":
            if e == 0: # train just once
                print("******************** Training ********************")
                print("Dataset: {} - Detection Algorithm: {}".format(dataset, DetectionAlgorithmsNames.GRE.value))
                # training always with greedy
                agent.env.set_communities(DetectionAlgorithmsNames.GRE.value)
                start_train_time = time.time()
                # Training
                agent.grid_search()
                end_train_time = time.time()
                train_time = end_train_time - start_train_time
                print(f"* Agent training time: {train_time}")

            for alg in detection_algs:
                editable_HyperParams.DETECTION_ALG_NAME = alg
                if alg not in results[dataset]:
                    results[dataset][alg] = dict()

                # ° ------    TEST    ------ ° #
                #elif args.mode == "test" or args.mode == "both":
                # To change the detection algorithm, or the dataset, on which the model
                # will be tested, please refer to the class HyperParams in the file
                # src/utils/utils.py, changing the values of the variables:
                # - GRAPH_NAME, for the dataset
                # - DETECTION_ALG, for the detection algorithm

                # To change the model path, please refer to the class FilePaths in the
                # file src/utils/utils.py
                model_path = FilePaths.TRAINED_MODEL.value

                # Tau defines the strength of the constraint on the goal achievement
                #taus = [0.3, 0.5, 0.8]
                taus = [0.5]
                # BETAs defines the number of actions to perform
                # Beta for the community hiding task defines the percentage of rewiring
                # action, add or remove edges
                #community_betas = [1, 3, 5, 10]
                # Beta for the node hiding task is a multiplier of mean degree of the
                # the graph
                node_betas = [0.5, 1, 2]
                #node_betas = [1]

                # Initialize the test class
                node_hiding = NodeHiding(agent=agent, model_path=model_path)
                #community_hiding = CommunityHiding(agent=agent, model_path=model_path)

                #print("* NOTE:")
                #print(
                #    "*    - Beta for Node Hiding is a multiplier of the mean degree of the graph"
                #)
                #print(
                    #"*    - Beta for Community Hiding is the percentage of rewiring action, add or remove edges"
                #)
                print("******************** Test ********************")
                print("Dataset: {} - Detection Algorithm: {}".format(dataset, alg))
                for tau in taus:
                    print("* Node Hiding with tau = {}".format(tau))
                    if f"tau {tau}" not in results[dataset][alg]:
                        results[dataset][alg][f"tau {tau}"] = dict()
                    for beta in node_betas:
                        if f"beta {beta}" not in results[dataset][alg][f"tau {tau}"]:
                            results[dataset][alg][f"tau {tau}"][f"beta {beta}"] = {
                                "sr": list(),
                                "nmi": list(),
                                "f1": list(),
                                "evading_time": list(),
                                "training_time": list()
                            }
                        print("* * Beta Node = {}".format(beta))
                        node_hiding.set_parameters(beta=beta, tau=tau)
                        node_hiding.run_experiment()
                        #print(f"* Agent test time: {node_hiding.agent_test_time}")

                        test_path=node_hiding.path_to_save+"evaluation_node_hiding.json"
                        with open(test_path , 'r') as file:
                            data = json.load(file)
                        success_var = data["Agent"]["goal"]
                        time_var = data["Agent"]["time"]
                        nmi_var = data["Agent"]["nmi"]
                        success_rate = sum(success_var)/len(success_var)
                        avg_evading_time = np.mean(time_var)
                        avg_nmi = np.mean(nmi_var)
                        f1 = (2*success_rate*avg_nmi)/(success_rate+avg_nmi)

                        results[dataset][alg][f"tau {tau}"][f"beta {beta}"]["sr"].append(success_rate)
                        results[dataset][alg][f"tau {tau}"][f"beta {beta}"]["nmi"].append(avg_nmi)
                        results[dataset][alg][f"tau {tau}"][f"beta {beta}"]["f1"].append(f1)
                        results[dataset][alg][f"tau {tau}"][f"beta {beta}"]["evading_time"].append(avg_evading_time)
                        results[dataset][alg][f"tau {tau}"][f"beta {beta}"]["training_time"].append(train_time)

    with open(FilePaths.TEST_DIR.value + 'results.json', 'w') as file:
        json.dump(results, file, indent=4)

    metrics_path = FilePaths.TEST_DIR.value + 'metrics.json'

    extrapolate_metrics(FilePaths.TEST_DIR.value + 'results.json',metrics_path,datasets,detection_algs,taus,node_betas)
    json_to_md_tables(metrics_path,"test/")
