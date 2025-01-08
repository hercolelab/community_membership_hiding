from src.utils.utils import HyperParams, Utils, FilePaths, DetectionAlgorithmsNames
from src.environment.graph_env import GraphEnvironment
from src.agent.agent import Agent

from src.utils.hiding_node import NodeHiding
from src.utils.hiding_community import CommunityHiding
from src.community_algs.dcmh.dcmh_hiding import DcmhHiding
from src.agent.agent import Agent

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
import random
import torch


log = logging.getLogger(__name__)

def set_seed(seed):
    log.info(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def exp(cfg: DictConfig, save_path: str, agent: Agent, wandb_cfg = None):

    set_seed(HyperParams.SEED.value)

    with wandb.init(config=wandb_cfg):
        wandb_cfg = wandb.config
        run_name = wandb.run.name
        sweep_path = save_path+f"/{run_name}"
        os.makedirs(sweep_path, exist_ok=True)
        
        log.info("Run name: {} - Dataset: {} - Detection Algorithm: {}".format(run_name,agent.env.env_name, agent.env.detection_alg))
        log.info(f"Output directory: {HydraConfig.get().runtime.output_dir}")
                
        model_path = FilePaths.TRAINED_MODEL.value

        tau = cfg["tau"]
        beta = cfg["beta"]

        #Candidate hyperparameters
        evader_cfg = cfg[f"{cfg['dataset']}"][f"testing_{cfg['test_alg']}"][f"tau_{cfg['tau']}"][f"beta_{cfg['beta']}"]
        evader_cfg['T'] = wandb_cfg.max_it
        evader_cfg['lr'] = wandb_cfg.lr
        evader_cfg['lambd'] = wandb_cfg.lambd
        log.info(f"Evader configuration: {evader_cfg}")

        # Initialize the test class
        node_hiding = NodeHiding(agent=agent, model_path=model_path, dcmh_config=cfg)
        log.info("* Node Hiding with tau = {}".format(tau))
        log.info("* * Beta Node = {}".format(beta))
        node_hiding.set_parameters(beta=beta, tau=tau, output_dir=sweep_path)
        node_hiding.run_experiment()

        # Open the JSON file for wandb log
        evaluation_path = f"{node_hiding.path_to_save}"+"evaluation_node_hiding.json"
        with open(evaluation_path, 'r') as eval_file:
            evaluation_data = json.load(eval_file)
        #Compute the F1 score
        f1 = [(2*goal*nmi)/(goal+nmi) for goal, nmi in zip(evaluation_data["DCMH"]["goal"], evaluation_data["DCMH"]["nmi"])]
        f1_score = np.mean(f1)
        log.info(f"* * F1 score: {f1_score}")
        wandb.log({"f1": f1_score})

        #Remove results to save space
        #os.remove(evaluation_path)
        #os.remove(f"{node_hiding.path_to_save}"+"dcmh_outputs.json")
        #shutil.rmtree(sweep_path)


@hydra.main(config_path="src/community_algs/dcmh/conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    save_path = HydraConfig.get().runtime.output_dir
    set_seed(HyperParams.SEED.value)

    dataset_names = {
        "kar": FilePaths.KAR.value,
        "words": FilePaths.WORDS.value,
        "vote": FilePaths.VOTE.value,
        "pow": FilePaths.POW.value,
        "fb": FilePaths.FB_75.value,
        "arxiv": FilePaths.ARXIV.value,
        "dblp": FilePaths.DBLP.value,
    }

    dataset = dataset_names[cfg["dataset"]]

    detection_algs = {
        "greedy": DetectionAlgorithmsNames.GRE.value,
        "louvain": DetectionAlgorithmsNames.LOUV.value,
        "walktrap": DetectionAlgorithmsNames.WALK.value,
    }

    algorithm = detection_algs[cfg["test_alg"]]

    # ° --- Environment Setup --- ° #
    env = GraphEnvironment(graph_path=dataset, community_detection_algorithm=algorithm)
    # ° ------  Agent Setup ----- ° #
    agent = Agent(env=env)


    sweep_config = {
        "method": "bayes",
        "metric": {"name": "f1", "goal": "maximize"},
        "parameters": {
            "max_it": {"values": list(range(50, 160,10))},
            "lr": {"min": 0.00001, "max": 0.01},
            "lambd": {"min": 0.1, "max": 100.0},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=f"dcmh_hyp_search {cfg['dataset']} {cfg['test_alg']} tau_{cfg['tau']} beta_{cfg['beta']}")
    wandb.agent(sweep_id, function=lambda: exp(cfg, save_path, agent), count=50)


if __name__ == "__main__":
    main()

