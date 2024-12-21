from src.utils.utils import editable_HyperParams, HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding
from src.community_algs.dcmh.dcmh_hiding import DcmhHiding

import argparse
import math
import time
import yaml
import hydra
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import wandb
import os
import json
import numpy as np
import shutil


dataset_names = {
    FilePaths.KAR.value: "kar",
    FilePaths.WORDS.value: "words",
    FilePaths.VOTE.value: "vote",
    FilePaths.NETS.value: "nets",
    FilePaths.POW.value: "pow",
    FilePaths.FB_75.value: "fb",
    FilePaths.ASTR.value: "astr",
}


def exp(cfg: DictConfig, save_path: str, wandb_cfg = None):

    datasets = [
        FilePaths.KAR.value if cfg["dataset"] == "kar" else
        FilePaths.WORDS.value if cfg["dataset"] == "words" else
        FilePaths.VOTE.value if cfg["dataset"] == "vote" else
        FilePaths.POW.value if cfg["dataset"] == "pow" else
        FilePaths.FB_75.value if cfg["dataset"] == "fb" else None
    ]
    detection_algs = [
        DetectionAlgorithmsNames.GRE.value if cfg["test_alg"] == "greedy" else
        DetectionAlgorithmsNames.LOUV.value if cfg["test_alg"] == "louvain" else
        DetectionAlgorithmsNames.WALK.value if cfg["test_alg"] == "walktrap" else None
    ]

    with wandb.init(config=wandb_cfg):
        wandb_cfg = wandb.config
        run_name = wandb.run.name
        sweep_path = save_path+f"/{run_name}"
        os.makedirs(sweep_path, exist_ok=True)

    

        for dataset in datasets:

            for alg in detection_algs:

                # ° --- Environment Setup --- ° #
                env = GraphEnvironment(graph_path=dataset, community_detection_algorithm=alg)
                cfg["dataset"] = dataset_names[dataset]
                # ° ------  Agent Setup ----- ° #
                agent = Agent(env=env)

                log.info("Run name: {} - Dataset: {} - Detection Algorithm: {}".format(run_name,dataset_names[dataset], alg))
                log.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
                agent.env.set_communities(alg)
                cfg["test_alg"] = alg

                # ° ------    TEST    ------ ° #
                
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
                taus = [cfg["tau"]]
                # BETAs defines the number of actions to perform
                # Beta for the node hiding task is a multiplier of mean degree of the
                # the graph
                node_betas = [cfg["beta"]]

                #Candidate hyperparameters
                evader_cfg = cfg[f"{cfg['dataset']}"][f"training_{cfg['train_alg']}"][f"testing_{cfg['test_alg']}"][f"tau_{cfg['tau']}"][f"beta_{cfg['beta']}"]
                evader_cfg['T'] = wandb_cfg.max_it
                evader_cfg['lr'] = wandb_cfg.lr
                evader_cfg['lambd'] = wandb_cfg.lambd
                log.info(f"Evader configuration: {evader_cfg}")

                # Initialize the test class
                node_hiding = NodeHiding(agent=agent, model_path=model_path, dcmh_config=cfg)

                for tau in taus:
                    log.info("* Node Hiding with tau = {}".format(tau))

                    for beta in node_betas:
                        log.info("* * Beta Node = {}".format(beta))
                        node_hiding.set_parameters(beta=beta, tau=tau, output_dir=sweep_path)
                        node_hiding.run_experiment()

                        # Open the JSON file for wandb log
                        evaluation_path = f"{node_hiding.path_to_save}"+"evaluation_node_hiding.json"
                        with open(evaluation_path, 'r') as eval_file:
                            evaluation_data = json.load(eval_file)
                        f1 = [(2*goal*nmi)/(goal+nmi) for goal, nmi in zip(evaluation_data["DCMH"]["goal"], evaluation_data["DCMH"]["nmi"])]
                        f1_score = np.mean(f1)
                        log.info(f"* * F1 score: {f1_score}")
                        wandb.log({"f1": f1_score})

                        #Remove results to save space
                        #os.remove(evaluation_path)
                        #os.remove(f"{node_hiding.path_to_save}"+"dcmh_outputs.json")
                        #shutil.rmtree(sweep_path)

log = logging.getLogger(__name__)
@hydra.main(config_path="src/community_algs/dcmh/conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    save_path = HydraConfig.get().runtime.output_dir

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "f1", "goal": "maximize"},
        "parameters": {
            "max_it": {"values": list(range(50, 110,10))},
            "lr": {"min": 0.001, "max": 0.5},
            "lambd": {"min": 0.001, "max": 5.0},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=f"dcmh_hyp_search {cfg['dataset']} {cfg['test_alg']} tau_{cfg['tau']} beta_{cfg['beta']}")
    wandb.agent(sweep_id, function=lambda: exp(cfg, save_path), count=60)


if __name__ == "__main__":
    main()

