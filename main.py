from src.utils.utils import editable_HyperParams, HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding

import argparse
import math
import time
import yaml
import hydra
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# Variables to choose the mode of the script
TRAIN = False
TEST = True

dataset_names = {
    FilePaths.KAR.value: "kar",
    FilePaths.WORDS.value: "words",
    FilePaths.VOTE.value: "vote",
    FilePaths.NETS.value: "nets",
    FilePaths.POW.value: "pow",
    FilePaths.FB_75.value: "fb",
    FilePaths.ASTR.value: "astr",
}


log = logging.getLogger(__name__)
@hydra.main(config_path="src/community_algs/dcmh/conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    datasets = [
        FilePaths.KAR.value,
        #FilePaths.WORDS.value,
        #FilePaths.VOTE.value,
        #FilePaths.POW.value,
        #FilePaths.FB_75.value,
    ]
    detection_algs = [
        DetectionAlgorithmsNames.GRE.value,
        #DetectionAlgorithmsNames.LOUV.value,
        #DetectionAlgorithmsNames.WALK.value,
    ]

    with open("src/community_algs/dcmh/conf/base.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    for dataset in datasets:
        # ° --- Environment Setup --- ° #
        env = GraphEnvironment(graph_path=dataset)
        cfg["dataset"] = dataset_names[dataset]

        # ° ------  Agent Setup ----- ° #
        agent = Agent(env=env)

        for alg in detection_algs:
            log.info("Dataset: {} - Detection Algorithm: {}".format(env.env_name, alg))
            log.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
            agent.env.set_communities(alg)
            cfg["test_alg"] = alg

            # ° ------    TRAIN    ------ ° #
            if TRAIN==True:
                # Training
                agent.grid_search()

            # ° ------    TEST    ------ ° #
            elif TEST==True:
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
                # Beta for the node hiding task is a multiplier of mean degree of the
                # the graph
                node_betas = [0.5, 1, 2]
                #node_betas = [1]

                # Initialize the test class
                node_hiding = NodeHiding(agent=agent, model_path=model_path, dcmh_config=cfg)

                log.info("* NOTE:")
                log.info(
                    "*    - Beta for Node Hiding is a multiplier of the mean degree of the graph"
                )
                for tau in taus:
                    log.info("* Node Hiding with tau = {}".format(tau))

                    for beta in node_betas:
                        log.info("* * Beta Node = {}".format(beta))
                        node_hiding.set_parameters(beta=beta, tau=tau, output_dir=HydraConfig.get().runtime.output_dir)
                        node_hiding.run_experiment()


    save_f1 = False
    if save_f1:
        Utils.plot_f1_all_datasets(
            datasets= [FilePaths.KAR.value,FilePaths.WORDS.value, FilePaths.VOTE.value, FilePaths.POW.value, FilePaths.FB_75.value],
            detection_algs= [ DetectionAlgorithmsNames.GRE.value, DetectionAlgorithmsNames.LOUV.value,DetectionAlgorithmsNames.WALK.value],
            #detection_algs= [ DetectionAlgorithmsNames.GRE.value, DetectionAlgorithmsNames.LOUV.value],
            #detection_algs= [ DetectionAlgorithmsNames.WALK.value],
            taus=[0.5],
            betas=[0.5,1,2],
            #betas=[0.5,1],
        )
    save_time = False
    if save_time:
        Utils.plot_time_all_datasets(
            datasets= [FilePaths.KAR.value,FilePaths.WORDS.value, FilePaths.VOTE.value, FilePaths.POW.value, FilePaths.FB_75.value],
            detection_algs= [ DetectionAlgorithmsNames.GRE.value, DetectionAlgorithmsNames.LOUV.value,DetectionAlgorithmsNames.WALK.value],
            #detection_algs= [ DetectionAlgorithmsNames.GRE.value, DetectionAlgorithmsNames.LOUV.value],
            #detection_algs= [ DetectionAlgorithmsNames.WALK.value],
            taus=[0.5],
            betas=[0.5,1,2],
            #betas=[0.5,1],
        )

if __name__ == "__main__":
    main()

            
