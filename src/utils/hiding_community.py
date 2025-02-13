from cgitb import reset
from re import S
from typing import List, Callable, Tuple

from sympy import degree_list
from src.utils.utils import HyperParams, Utils, FilePaths
from src.environment.graph_env import GraphEnvironment
from src.community_algs.metrics.deception_score import DeceptionScore

from src.community_algs.baselines.community_hiding.sadden import Safeness

# from src.community_algs.baselines.community_hiding.safeness import Safeness
from src.community_algs.baselines.community_hiding.modularity import Modularity

from src.agent.agent import Agent

from tqdm import trange
import networkx as nx
import cdlib
import time
import copy
import math


class CommunityHiding:
    """
    Class to evaluate the performance of the agent in the community hiding task,
    where the agent has to hide a community from a detection algorithm.
    Futhermore, it is compared with other baselines algorithms:
        - Safeness Community Deception
    """

    def __init__(
        self,
        agent: Agent,
        model_path: str,
        lr: float = HyperParams.LR_EVAL.value,
        gamma: float = HyperParams.GAMMA_EVAL.value,
        lambda_metric: float = HyperParams.LAMBDA_EVAL.value,
        alpha_metric: float = HyperParams.ALPHA_EVAL.value,
        epsilon_prob: float = HyperParams.EPSILON_EVAL.value,
        eval_steps: int = HyperParams.STEPS_EVAL.value,
    ) -> None:
        self.agent = agent
        self.original_graph = agent.env.original_graph.copy()
        self.model_path = model_path
        self.env_name = agent.env.env_name
        self.detection_alg = agent.env.detection_alg
        self.community_target = agent.env.community_target

        # Copy the community structure to avoid modifying the original one
        self.community_structure = copy.deepcopy(agent.env.original_community_structure)
        # self.node_target = agent.env.node_target

        self.lr = lr
        self.gamma = gamma
        self.lambda_metric = lambda_metric
        self.alpha_metric = alpha_metric
        self.epsilon_prob = epsilon_prob
        self.eval_steps = eval_steps

        self.beta = None
        self.tau = None
        self.edge_budget = None
        self.max_steps = None
        self.community_edge_budget = None

        self.safeness_obj = None
        self.modularity_obj = None
        self.deception_score_obj = None

        self.evaluation_algs = ["Agent", "Safeness", "Modularity"]

        # Use a list to store the beta values already computed, beacuse the
        # Community Deception algorithms are not influenced by the value of
        # tau, so we can compute the beta values only once
        self.beta_values_computed = []
        # Use a dict to store the results of the Community Deception algorithms
        # for each beta value
        self.beta_values_results = dict()

    def set_parameters(self, beta: int, tau: float, reset=True) -> None:
        """Set the environment with the new parameters, for new experiments

        Parameters
        ----------
        beta : int
            In this case beta is the percentage of edges to remove or add
        tau : float
            Constraint on the goal achievement
        """
        self.beta = beta
        self.tau = tau

        # Set community beta value as key of the dictionary
        if self.beta not in self.beta_values_results:
            self.beta_values_results[self.beta] = dict()

        self.agent.env.tau = tau
        # ! NOTE: It isn't the same beta as the one used in the Node Hiding task
        # self.agent.env.beta = beta
        # self.agent.env.set_rewiring_budget()

        # Budget for the whole community, beta percentage of the number of nodes
        # in the target community
        self.community_edge_budget = self.beta
        # self.community_edge_budget = math.ceil(
        #    len(self.community_target) * (self.beta / 100)
        # )

        # Set the node budge as the community budget
        # self.node_edge_budget = self.community_edge_budget

        # We can't call the set_rewiring_budget function because we don't have
        # the beta value multiplier, and also we need to adapt to the Community
        # Hiding task, where the budget for the agent is set as the BETA percentage
        # of all the edges in the graph divided by the number of nodes in the
        # target community. So we set manually all the values of set_rewiring_budget
        # function.
        # self.agent.env.edge_budget = self.node_edge_budget
        # self.agent.env.max_steps = self.node_edge_budget * HyperParams.MAX_STEPS_MUL.value
        self.agent.env.used_edge_budget = 0
        self.agent.env.stop_episode = False
        self.agent.env.reward = 0
        self.agent.env.old_rewards = 0
        self.agent.env.possible_actions = self.agent.env.get_possible_actions()
        self.agent.env.len_add_actions = len(self.agent.env.possible_actions["ADD"])

        # Initialize the log dictionary
        if reset:
            self.set_log_dict()

        self.path_to_save = (
            FilePaths.TEST_DIR.value
            + f"{self.env_name}/{self.detection_alg}/"
            + f"community_hiding/"
            + f"tau_{self.tau}/"
            + f"beta_{self.beta}/"
            # + f"eps_{self.epsilon_prob}/"
            # + f"lr_{self.lr}/gamma_{self.gamma}/"
            # + f"lambda_{self.lambda_metric}/alpha_{self.alpha_metric}/"
        )

    def reset_experiment(self) -> None:
        """
        Reset the environment and the agent at the beginning of each episode,
        and change the target community and node
        """
        self.agent.env.change_target_community()

        # Copy the community target to avoid modifying the original one
        self.community_target = copy.deepcopy(self.agent.env.community_target)
        # self.node_target = self.agent.env.node_target

        self.set_parameters(self.beta, self.tau, reset=False)

        # Initialize the Deception Score algorithm
        self.deception_score_obj = DeceptionScore(copy.deepcopy(self.community_target))

        self.safeness_obj = Safeness(
            self.community_edge_budget,
            self.original_graph,
            self.community_target,
            self.community_structure,
        )

        self.modularity_obj = Modularity(
            self.community_edge_budget,
            self.original_graph,
            self.community_target,
            self.community_structure,
            self.agent.env.detection,
        )

        # # Compute a Dictionary where the keys are the nodes of the community
        # # target and the values are the centrality of the nodes
        # node_centralities = nx.centrality.degree_centrality(self.original_graph)
        # # Get the subset of the dictionary with only the nodes of the community
        # node_com_centralities = {k: node_centralities[k] for k in self.community_target}
        # # Order in descending order the dictionary
        # self.node_com_centralities = dict(
        #     sorted(
        #         node_com_centralities.items(), key=lambda item: item[1], reverse=True
        #     )
        # )

        # ! Compute the budget for each node in the target community, for the
        # function run_agent_distributed_budget()
        # self.compute_budget_proportionally(self.original_graph, self.community_target)

    def compute_budget_proportionally(
        self, graph: nx.Graph, community_target: List[int]
    ) -> None:
        """
        Compute the budget for each node in the target community, proportionally
        to the degree of each node.

        Parameters
        ----------
        graph : nx.Graph
            Graph on which the agent is acting
        community_target : List[int]
            Target community
        """
        # Calculate the total degree of all nodes in the graph
        total_degree = sum(dict(graph.degree()).values())
        remaining_budget = self.community_edge_budget
        self.budget_per_node = {}

        if total_degree == 0:
            # Divide the budget equally between all nodes
            budget_per_node = self.community_edge_budget // len(community_target)
            self.budget_per_node = {node: budget_per_node for node in community_target}
            return

        # Order the nodes in descending order based on their degree
        sorted_nodes = sorted(
            community_target, key=lambda node: graph.degree(node), reverse=True
        )

        for node in sorted_nodes:
            degree = graph.degree(node)
            proportion = degree / total_degree
            new_budget = min(
                self.community_edge_budget,
                math.ceil(self.community_edge_budget * proportion),
            )
            self.budget_per_node[node] = new_budget
            remaining_budget -= new_budget

    def compute_budget_betweenness(
        self, graph: nx.Graph, community_target: List[int], k=3
    ) -> None:
        """
        Compute the budget for each node in the target community, proportionally
        to the betweenness centrality of each node.

        Parameters
        ----------
        graph : nx.Graph
            Graph on which the agent is acting
        community_target : List[int]
            Target community
        k : int, optional
            Number of nodes to allocate the budget, by default 3
        """
        betweenness = nx.betweenness_centrality(graph)
        # Order the nodes in descending order based on their degree of centrality
        sorted_nodes = sorted(
            community_target, key=lambda node: betweenness[node], reverse=True
        )
        
        # Calculate the total degree of centrality of the first k nodes of the community
        total_betweenness = sum(betweenness[node] for node in sorted_nodes[:k])

        remaining_budget = self.community_edge_budget
        self.budget_per_node = {}

        for node in sorted_nodes[:k]:
            centrality = betweenness[node]
            proportion = centrality / total_betweenness
            new_budget = min(
                self.community_edge_budget,
                math.ceil(self.community_edge_budget * proportion),
            )
            self.budget_per_node[node] = new_budget
            remaining_budget -= new_budget
        
        for node in sorted_nodes[k:]:
            self.budget_per_node[node] = 0

    def run_experiment(self) -> None:
        # Start evaluation
        preferred_size_list = HyperParams.PREFERRED_COMMUNITY_SIZE.value
        sizes = trange(
            len(preferred_size_list), desc="* * * Community Size", leave=True
        )
        for i in sizes:
            self.agent.env.set_preferred_community_size(preferred_size_list[i])
            compute_baselines = True
            self.reset_experiment()

            # Print community size in tqdm
            sizes.set_description(f"* * * Community Size: {len(self.community_target)}")

            steps = trange(self.eval_steps, desc="* * * * Testing Episode", leave=False)
            for step in steps:
                # self.compute_budget_proportionally(
                #     self.original_graph, self.community_target
                # )
                self.compute_budget_betweenness(
                    self.original_graph, self.community_target
                )
                # ° ------ Agent Rewiring ------ ° #
                steps.set_description(
                    f"* * * * Testing Episode {step+1} | Agent Rewiring"
                )
                # self.run_alg(self.run_agent)
                self.run_alg(self.run_agent_distributed_budget)

                # ° --------- Baselines --------- ° #
                if compute_baselines:
                    # Safeness
                    steps.set_description(
                        f"* * * * Testing Episode {step+1} | Safeness Rewiring"
                    )
                    self.run_alg(self.run_safeness)

                    # Modularity
                    steps.set_description(
                        f"* * * * Testing Episode {step+1} | Modularity Rewiring"
                    )
                    self.run_alg(self.run_modularity)
                    compute_baselines = False

        Utils.check_dir(self.path_to_save)
        Utils.save_test(
            log=self.log_dict,
            files_path=self.path_to_save,
            log_name="evaluation_community_hiding",
            algs=self.evaluation_algs,
            metrics=["nmi", "goal", "deception_score", "time", "steps"],
        )

    def run_alg(self, function: Callable) -> None:
        """
        Wrapper function to run the evaluation of a generic algorithm

        Parameters
        ----------
        function : Callable
            Algorithm to evaluate
        """
        start = time.time()
        alg_name, goal, nmi, deception_score, step = function()
        end = time.time() - start
        # Save results in the log dictionary
        self.save_metrics(alg_name, goal, nmi, deception_score, end, step)

    ############################################################################
    #                               AGENT                                      #
    ############################################################################
    def run_agent(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the agent on the Node Hiding task. In this case the agent starts
        to hide the node with the highest centrality in the target community, and
        with the budget equal to the Community Deception baselines, and it is
        scaled down at each step, based on the number of steps performed.

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        tot_steps = 0
        agent_goal_reached = False
        # Initialize the new community structure as the original one, because
        # the agent could not perform any rewiring
        communities = self.community_structure
        # As first node to hide, we choose the node with the highest centrality
        # in the target community
        node = self.node_com_centralities.popitem()[0]
        while True:
            self.agent.env.node_target = node
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = self.agent.test(
                lr=self.lr,
                gamma=self.gamma,
                lambda_metric=self.lambda_metric,
                alpha_metric=self.alpha_metric,
                epsilon_prob=self.epsilon_prob,
                model_path=self.model_path,
            )
            # Get the new community structure
            self.agent.env.new_community_structure = (
                self.agent.env.detection.compute_community(new_graph)
            )
            new_communities = self.agent.env.new_community_structure
            # Check if the agent performed any rewiring
            if new_communities is None:
                new_communities = communities
            # Get the community in the new community structure, which contains
            # the highest number of nodes of the target community
            new_community = max(
                new_communities.communities,
                key=lambda c: sum(1 for n in self.community_target if n in c),
            )
            # Recompute the node centralities after the rewiring
            node_centralities = nx.centrality.degree_centrality(new_graph)
            # Choose the next node to hide, as the node with the highest
            # centrality in the new community
            if self.agent.env.used_edge_budget > 0:
                # If the agent has not performed all the rewiring actions
                node = max(
                    (n for n in new_community if n in self.community_target),
                    key=lambda n: node_centralities[n],
                )
            tot_steps += self.agent.env.used_edge_budget
            # Reduce the edge budget
            self.agent.env.edge_budget = self.node_edge_budget - tot_steps
            self.agent.env.max_steps = (
                self.agent.env.edge_budget * HyperParams.MAX_STEPS_MUL.value
            )
            # Check if the agent reached the goal
            if tot_steps >= self.community_edge_budget or node is None:
                if self.agent.env.new_community_structure is None:
                    # The agent did not perform any rewiring, i.e. are the same communities
                    agent_goal_reached = False
                    break
                if (
                    self.community_target
                    not in self.agent.env.new_community_structure.communities
                ):
                    agent_goal_reached = True
                communities = self.agent.env.new_community_structure
                break

        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(
            self.community_structure, self.agent.env.new_community_structure
        )
        goal = 1 if agent_goal_reached else 0
        return self.evaluation_algs[0], goal, nmi, deception_score, tot_steps

    def run_agent_distributed_budget(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the agent on the Node Hiding task. In this case the budget is
        distributed proportionally to the degree of each node in the target
        community, and the agent starts to hide the node with the highest budget.

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        tot_steps = 0
        agent_goal_reached = False
        # Choose the node from the target community with the highest budget
        node = max(
            self.community_target, key=lambda n: self.budget_per_node[n], default=None
        )

        if node is None:
            print(self.budget_per_node)
            print(self.community_edge_budget)
            for node in self.community_target:
                print(node, self.original_graph.degree(node))
            raise Exception("Node is None")

        while True:
            self.agent.env.node_target = node
            # Set the agent edge budget as the budget of the node
            self.agent.env.edge_budget = self.budget_per_node[node]
            # Set Max Steps as the budget of the node multiplied by a constant
            self.agent.env.max_steps = (
                self.community_edge_budget * HyperParams.MAX_STEPS_MUL.value
            )
            # The agent possible action are changed in the test function, which
            # calls the reset function of the environment
            new_graph = self.agent.test(
                lr=self.lr,
                gamma=self.gamma,
                lambda_metric=self.lambda_metric,
                alpha_metric=self.alpha_metric,
                epsilon_prob=self.epsilon_prob,
                model_path=self.model_path,
            )

            # TEST: decrease the budget of the node
            self.budget_per_node[node] -= self.agent.env.used_edge_budget
            node = max(
                # (n for n in self.community_target if n in new_community),
                (n for n in self.community_target),
                key=lambda n: self.budget_per_node[n],
                default=None,
            )
            # Increment the total steps
            tot_steps += self.agent.env.used_edge_budget

            if tot_steps >= self.community_edge_budget or node is None:
                break

        # Compute new community structure
        self.agent.env.new_community_structure = (
            self.agent.env.detection.compute_community(new_graph)
        )

        if (
            self.community_structure
            not in self.agent.env.new_community_structure.communities
        ):
            agent_goal_reached = True
        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(self.agent.env.new_community_structure.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(
            self.community_structure, self.agent.env.new_community_structure
        )
        goal = 1 if agent_goal_reached else 0
        return self.evaluation_algs[0], goal, nmi, deception_score, tot_steps

    ############################################################################
    #                               BASELINES                                  #
    ############################################################################
    def run_safeness(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the Safeness algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        new_graph, steps = self.safeness_obj.run()

        # Compute the new community structure
        new_communities = self.agent.env.detection.compute_community(new_graph)

        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(new_communities.communities),
        )
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, new_communities)
        goal = 1 if self.community_target not in new_communities.communities else 0
        return self.evaluation_algs[1], goal, nmi, deception_score, steps

    def run_modularity(self) -> Tuple[str, int, float, float, int]:
        """
        Evaluate the Safeness algorithm on the Node Hiding task

        Returns
        -------
        Tuple[str, nx.Graph, int, float, int]
            Algorithm name, goal, nmi, deception score, steps
        """
        new_graph, steps, new_communities = self.modularity_obj.run()

        # Compute Deception Score between the new community structure and the
        # original one
        deception_score = self.deception_score_obj.get_deception_score(
            new_graph.copy(),
            copy.deepcopy(new_communities.communities),
        )
        # print("Deception Score:", deception_score)
        # Compute NMI between the new community structure and the original one
        nmi = self.get_nmi(self.community_structure, new_communities)
        goal = 1 if self.community_target not in new_communities.communities else 0
        return self.evaluation_algs[2], goal, nmi, deception_score, steps

    ############################################################################
    #                               UTILS                                      #
    ############################################################################
    def get_nmi(
        self,
        old_communities: cdlib.NodeClustering,
        new_communities: cdlib.NodeClustering,
    ) -> float:
        """
        Compute the Normalized Mutual Information between the old and the new
        community structure

        Parameters
        ----------
        old_communities : cdlib.NodeClustering
            Community structure before deception
        new_communities : cdlib.NodeClustering
            Community structure after deception

        Returns
        -------
        float
            Normalized Mutual Information between the old and the new community
        """
        if new_communities is None:
            # The agent did not perform any rewiring, i.e. are the same communities
            return 1
        return old_communities.normalized_mutual_information(new_communities).score

    ############################################################################
    #                               LOG                                        #
    ############################################################################
    def set_log_dict(self) -> None:
        self.log_dict = dict()

        for alg in self.evaluation_algs:
            self.log_dict[alg] = {
                "goal": [],
                "nmi": [],
                "time": [],
                "deception_score": [],
                "steps": [],
                "community_len": [],
            }

        # Add environment parameters to the log dictionaryù
        self.log_dict["env"] = dict()
        self.log_dict["env"]["dataset"] = self.env_name
        self.log_dict["env"]["detection_alg"] = self.detection_alg
        self.log_dict["env"]["beta"] = self.beta
        self.log_dict["env"]["tau"] = self.tau
        self.log_dict["env"]["edge_budget"] = self.edge_budget
        self.log_dict["env"]["max_steps"] = self.max_steps

        # Add Agent Hyperparameters to the log dictionary
        self.log_dict["Agent"]["lr"] = self.lr
        self.log_dict["Agent"]["gamma"] = self.gamma
        self.log_dict["Agent"]["lambda_metric"] = self.lambda_metric
        self.log_dict["Agent"]["alpha_metric"] = self.alpha_metric
        self.log_dict["Agent"]["epsilon_prob"] = self.epsilon_prob

    def save_metrics(
        self,
        alg: str,
        goal: int,
        nmi: float,
        deception_score: float,
        time: float,
        steps: int,
    ) -> dict:
        """Save the metrics of the algorithm in the log dictionary"""
        self.log_dict[alg]["goal"].append(goal)
        self.log_dict[alg]["nmi"].append(nmi)
        self.log_dict[alg]["deception_score"].append(deception_score)
        self.log_dict[alg]["time"].append(time)
        self.log_dict[alg]["steps"].append(steps)
        self.log_dict[alg]["community_len"].append(len(self.community_target))
