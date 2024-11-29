from src.utils.utils import HyperParams, editable_HyperParams, Utils, editable_FilePaths, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding

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
    parser = argparse.ArgumentParser(description="PyTorch A2C")
    # Mode: train or test
    parser.add_argument("--mode", type=str, default="both", help="train | test | both")
    # Argument parsing
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    datasets = [
        FilePaths.KAR.value,
        #FilePaths.WORDS.value,
        #FilePaths.VOTE.value,
        #FilePaths.NETS.value,
        #FilePaths.POW.value,
        #FilePaths.FB_75.value,
        #FilePaths.ASTR.value,
    ]
    datasets_names = {
        FilePaths.KAR.value: "karate",
        FilePaths.WORDS.value: "Words",
        FilePaths.VOTE.value: "vote",
        FilePaths.NETS.value: "nets",
        FilePaths.POW.value: "pow",
        FilePaths.FB_75.value: "fb-75",
        FilePaths.ASTR.value: "astr",
    }
    detection_algs = [
        DetectionAlgorithmsNames.GRE.value,
        #DetectionAlgorithmsNames.LOUV.value,
        #DetectionAlgorithmsNames.WALK.value,
    ]

    seed = int(time.time())

    for dataset in datasets:
        editable_HyperParams.GRAPH_NAME = dataset
        # ° --- Environment Setup --- ° #
        env = GraphEnvironment(seed=seed, graph_path=dataset)

        # ° ------  Agent Setup ----- ° #
        agent = Agent(env=env)

        # ° ------    TRAIN    ------ ° #
        train_alg = DetectionAlgorithmsNames.GRE.value
        if args.mode == "train" or args.mode == "both":
            print("******************** Training ********************")
            # training with one algorithm 
            print("Dataset: {} - Detection Algorithm: {}".format(datasets_names[dataset], train_alg))
            agent.env.set_communities(train_alg)
            start_train_time = time.time()
            # Training
            agent.grid_search(datasets_names[dataset], train_alg)
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            print(f"* Agent training time: {train_time}")

        # ° ------    TEST    ------ ° #
        elif args.mode == "test" or args.mode == "both":
        # To change the detection algorithm, or the dataset, on which the model
        # will be tested, please refer to the class editableHyperParams in the file
        # src/utils/utils.py, changing the values of the variables:
        # - GRAPH_NAME, for the dataset
        # - DETECTION_ALG, for the detection algorithm

            # To change the model path, please refer to the class editable_FilePaths in the
            # file src/utils/utils.py
            editable_FilePaths.TRAINED_MODEL = "src/models/steps-10000_"+datasets_names[dataset]+"_"+train_alg+"_model.pth"
            model_path = editable_FilePaths.TRAINED_MODEL

            for alg in detection_algs:
                editable_HyperParams.DETECTION_ALG_NAME = alg
                agent.env.set_communities(alg)

                # Tau defines the strength of the constraint on the goal achievement
                #taus = [0.3, 0.5, 0.8]
                taus = [0.5]
                # BETAs defines the number of actions to perform
                # Beta for the node hiding task is a multiplier of mean degree of the
                # the graph
                node_betas = [0.5, 1, 2]
                #node_betas = [1]

                # Initialize the test class
                node_hiding = NodeHiding(agent=agent, model_path=model_path)

                print("******************** Test ********************")
                print("Dataset: {} - Detection Algorithm: {}".format(datasets_names[dataset], alg))
                for tau in taus:
                    print("* Node Hiding with tau = {}".format(tau))

                    for beta in node_betas:
                        print("* * Node Hiding with beta = {}".format(beta))
                        node_hiding.set_parameters(beta=beta, tau=tau)
                        node_hiding.run_experiment()

        else:
                raise ValueError(
                    "Invalid mode. Please choose between 'train' and 'test'"
                )

