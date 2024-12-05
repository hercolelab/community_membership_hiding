from src.environment.graph_env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils, editable_HyperParams
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from cdlib import algorithms
from typing import List, Callable, Tuple
import random
import time
import copy
import numpy as np
import igraph as ig
from scipy.stats import rankdata
import torch
import torch.optim as optim
from omegaconf import DictConfig

class DcmhHiding():
    def __init__(
        self, 
        env: GraphEnvironment, 
        steps: int,
    ):
        self.env = env
        self.graph = self.env.original_graph
        self.detection_alg = self.env.detection
    ############################################################################
    #                                EVADING                                   #
    ############################################################################

    def comm_evading(self, cfg: DictConfig) -> Tuple[ig.Graph, int]:

        # Parameters
        T, lr, u, lambd, budget, tau, attention, reinit = self.get_evader_parameters(cfg)
        seed = editable_HyperParams.seed

        # Training detection algorithm
        da_train = CommunityDetectionAlgorithm("greedy")

        # Evasion parameters
        t = 0
        budget_used = 0
        goal = 0

        # Network
        G = da_train.networkx_to_igraph(self.graph)
        n_nodes = G.vcount()
        g_prime = G.copy()
        neighbors = torch.LongTensor(G.neighbors(u))
        a_u = torch.zeros(G.vcount(), dtype=torch.int)
        a_u[neighbors] = 1
        history = [a_u]
        temp_cf = {'counterfactual': g_prime, 'steps': 0}
        fixed_nodes = torch.LongTensor([v for v in neighbors if G.degree(v) == 1]+[v for v in neighbors if G.degree(u) == 1])
        edges_changed = {}
        #Communities
        old_communities = da_train.compute_community(G, dcmh=True)
        old_community_u = self.get_new_community(old_communities, u)
        new_communities = copy.deepcopy(old_communities)

        #Perturbation vector
        """We generate random vector s.t. threshold(tanh(x_hat)) = 0 """
        x_hat, optimizer = self.initialize_perturbation_vector(n_nodes, lr, u, fixed_nodes)

        # Candidate list for the loss
        v_opt = self.candidate_list(G,old_communities,u,attention)
        v_opt[u] = torch.Tensor([0])
        v_opt[fixed_nodes] = torch.Tensor([1])

        #EVASION LOOP
        while (goal==0 or budget_used < budget) and t < T:
            
            #Perturbation update
            p_hat = torch.tanh(x_hat)
            p = self.threshold_tanh(p_hat.detach().clone(), 0.5, -0.5)
            a_new = self.clamp(torch.Tensor(a_u + p))
            history.append(a_new)

            edges_changed, n_changes = self.get_changes(history[-2], history[-1], u)
            if n_changes > 0:
                budget_used += n_changes
                edge_list = g_prime.get_edgelist()
                updated_edge_list = self.update_edge_list(edge_list,edges_changed)
                g_prime = ig.Graph(n=G.vcount(), edges=updated_edge_list)
                new_communities = da_train.compute_community(g_prime, dcmh=True)
                new_community_u = self.get_new_community(new_communities, u)
                goal = self.check_goal(old_community_u,new_community_u,u,tau)
                n_changes = 0 #reset changes
                

            l_decept = self.loss_decept(a_u, p_hat, v_opt, self.frobenius_dist)
            l_dist = self.loss_dist(p_hat)
            loss = l_decept + lambd * l_dist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1
            
            if budget_used > budget:
                if reinit: 
                    x_hat, optimizer = self.initialize_perturbation_vector(n_nodes, lr, u, fixed_nodes) 
                    # Restore parameters for evasion loop
                    goal = 0 
                    budget_used=0
                    history.append(a_u)
                    g_prime = G.copy()
                else: 
                    g_prime = temp_cf['counterfactual'].copy()
                    break
        
            if budget_used == budget:
                temp_cf['counterfactual'] = g_prime.copy()
                temp_cf['steps'] = budget_used
                if goal == 0:
                    if reinit: 
                        x_hat, optimizer = self.initialize_perturbation_vector(n_nodes, lr, u, fixed_nodes)
                        # Restore parameters for evasion loop
                        budget_used=0
                        history.append(a_u)
                        g_prime = G.copy()
                    else: 
                        break

        if goal == 1:
            return g_prime, budget_used
        else:
            return temp_cf['counterfactual'], temp_cf['steps']
    
    ############################################################################
    #                                  LOSS                                    #
    ############################################################################

    def loss_decept(self, a_u, p_hat, v_opt, dist_f):
        """Compute the decept loss as the distance between the current adjacencies of the target node u and a target list L_out.
        
        Parameters
        ----------
        a_u: List
            Adjacency list of node u extracted from A.
        p_hat: List
            Parameters list to optimize.
        L_out: List
            List of target nodes that u should attach with.
        fixed_nodes: List
            List of fixed nodes for the perturbation
        dist_f: List
            Distance function to employ in the computation of the loss.
            
        Returns
        -------
        loss: Float
            The value of the deception loss.
        """
        d = dist_f(v_opt,a_u+p_hat)**2
        return d

    def loss_dist(self, p):
        """Compute the distance loss as the norm of the perturbation
        
        Parameters
        ----------
        p: Tensor
            Perturbation vector
            
        Returns
        -------
        loss: float
            The value of the distance loss.
        """
        dg = torch.norm(p)
    
        return dg
    
    ############################################################################
    #                                  UTILS                                   #
    ############################################################################

    def check_goal(self, old_community: List[int], new_community: List[int],u : int, tau: float) -> int:
        """
        Check if the goal of hiding the target node was achieved

        Parameters
        ----------
        new_community : int
            New community of the target node

        Returns
        -------
        int
            1 if the goal was achieved, 0 otherwise
        """
        if len(new_community) == 1:
            return 1
        # Copy the communities to avoid modifying the original ones
        new_community_copy = new_community.copy()
        new_community_copy.remove(u)
        old_community_copy = old_community.copy()
        old_community_copy.remove(u)
        # Compute the similarity between the new and the old community
        similarity = self.env.community_similarity(
            new_community_copy, old_community_copy
        )
        del new_community_copy, old_community_copy
        if similarity <= tau:
            return 1
        return 0

    def candidate_list(self, G: ig.Graph , communities: List[List[int]], u: int, attention: bool) -> torch.Tensor:
        """
        Generate the candidate list.

        Parameters
        ----------
        G
            The graph.
        communities
            The communities.
        u
            The target node.
        attention
            Whether to use attention.
        """
        n = G.vcount()
        L = torch.ones(n)
        L_in = torch.LongTensor(self.get_new_community(communities, u))
        L[L_in] = torch.Tensor([0])
        if attention is True:
            att = torch.Tensor(self.compute_attention(G,communities,u))*0.5
            v_opt = torch.where(L == 1, 0.5 + att, 0.5 - att)
            return v_opt
        else: 
            return L
    
    def get_new_community(
                self,
                new_community_structure: List[List[int]],
                u: int) -> List[int]:
        """
        Search the community target in the new community structure after 
        deception. As new community target after the action, we consider the 
        community that contains the target node, if this community satisfies 
        the deception constraint, the episode is finished, otherwise not.

        Parameters
        ----------
        node_target : int
            Target node to be hidden from the community
        new_community_structure : List[List[int]]
            New community structure after deception

        Returns
        -------
        List[int]
            New community target after deception
        """
        # if new_community_structure is None:
        #     # The agent did not perform any rewiring, i.e. are the same communities
        #     return self.target_community
        for community in new_community_structure.communities:
            if u in community:
                return community
        raise ValueError("Community not found")
    
    def compute_attention(self, G: ig.Graph, communities: List[List[int]], u: int) -> np.ndarray:
        """
        Compute the importance scores.
        
        Parameters
        ----------
        G
            The graph.
        communities
            The communities.
        u
            The target node.
        
        Returns
        -------
        torch.Tensor
            The averaged importance scores.
        """

        #Centrality score -- attention1
        centrality = np.array(G.betweenness(directed=False))
        ranks = rankdata(centrality, method='average')
        att1 = (ranks - 1) / (len(ranks) - 1)
        
        #Degree score -- attention2
        degrees = np.array(G.degree())
        ranks = rankdata(degrees, method='average')
        att2 = (ranks - 1) / (len(ranks) - 1)

        #Inter-intra community score -- attention3
        att3a = np.zeros(G.vcount())
        for c in communities.communities:
            subgraph = G.subgraph(c)
            degrees = np.array(subgraph.degree())
            ranks = rankdata(degrees, method='average')
            scaled_degrees = (ranks - 1) / (len(ranks) - 1) 
            for node, scaled_degree in zip(c, scaled_degrees):
                att3a[node] = scaled_degree
            

        att3b = np.zeros(G.vcount())
        community_nodes = self.get_new_community(communities, u)
        inter_c_nodes = sorted(set(range(G.vcount())) - set(community_nodes) - set(G.neighbors(u)))
        subgraph = G.subgraph(inter_c_nodes)
        degrees = np.array(subgraph.degree())
        ranks = rankdata(degrees, method='average')
        scaled_degrees = (ranks - 1) / (len(ranks) - 1) 
        for node, scaled_degree in zip(inter_c_nodes, scaled_degrees):
            att3b[node] = scaled_degree
        att3 = (att3a + att3b)/2

        #Attention by aggregation
        return (att1 + att2 + att3)/3
    
    def initialize_perturbation_vector(
            self,
            n_nodes: int, 
            lr: float, 
            u: int, 
            fixed_nodes: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.optim.Optimizer]:
        """
        Initialize the perturbation vector s.t. threshold(tanh(x_hat)) = 0.

        Parameters
        ----------
        n_nodes
            The number of nodes.
        lr
            The learning rate.
        u
            The target node.
        fixed_nodes
            The fixed nodes, e.g. neighbours with degree 1.
        
        Returns
        -------
        x_hat
            The perturbation vector.
        optimizer
            The optimizer.
        """
        torch.seed()
        x_hat = (2*torch.rand(n_nodes) - 1)*0.5
        x_hat[u] = torch.Tensor([0])
        x_hat[fixed_nodes] = torch.Tensor([0])
        """
        Possibility to add memory property on the algorithm
        --- TO THINK ABOUT IT ---
        """
        """
        #changed_nodes = set()
        #for change_type in changes.values():
            #for node_pair in change_type:
                #changed_nodes.update(node_pair)
        #changed_nodes.discard(u)
        #x_hat[list(changed_nodes)] = torch.Tensor([0])
        #log.info(torch.tanh(x_hat))
        """
        x_hat = x_hat.requires_grad_(True)
        optimizer = optim.Adam([x_hat], lr=lr)
        return x_hat,optimizer
    
    def get_evader_parameters(self, cfg: DictConfig):
        """
        Get the evader parameters from the configuration file.

        Parameters
        ----------
        cfg
            The configuration file.
        
        Returns
        -------
        T
            The number of iterations.
        lr
            The learning rate.
        u
            The target node.
        lambd
            The lambda parameter.
        tau
            The similarity threshold.
        beta_factor
            The budget factor.
        attention
            Whether to use attention.
        reinit
            Whether to reinitialize the perturbation vector.
        dist
            The distance function.
        """
        evader_cfg = self.get_evader_configuration(cfg)
        T = evader_cfg["T"]
        lr = evader_cfg["lr"]
        u = evader_cfg["u"]
        lambd = evader_cfg["lambd"]
        budget = evader_cfg["budget"]
        tau = evader_cfg["tau"]
        attention = evader_cfg["attention"]
        reinit = evader_cfg["reinitialization"]
        return T, lr, u, lambd, budget, tau, attention, reinit

    def get_evader_configuration(self, cfg: DictConfig):
        """
        Get the evader configuration from the configuration file.

        Parameters
        ----------
        cfg
            The general configuration file.
        
        Returns
        -------
        evader_cfg
            The evader configuration for a specific dataset.
        """
        dataset = cfg["dataset"]
        if dataset == 'karate':
            evader_cfg = cfg["karate"]["evader"]
        elif dataset == 'words':
            evader_cfg = cfg["words"]["evader"]
        elif dataset == 'vote':
            evader_cfg = cfg["vote"]["evader"]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return evader_cfg
    
    def get_changes(self, a1: torch.Tensor, a2: torch.Tensor, u: int) -> Tuple[dict, int]:
        """
        Get the changes in the u-th adjacency vector.

        Parameters
        ----------
        a1
            First adjacency vector.
        a2 
            Second adjacency vector.
        u
            Target node.

        Returns
        -------
        changes
            Dictionary containing the changes in the adjacency vector.
        
        count
            Number of changes in the adjacency vector.
        """
        changes = {"added": [], "removed": []}
        count = 0
        done = set()
        for i in range(len(a1)):
            if a1[i] != a2[i] and ((min(u, i), max(u, i))) not in done:
                count += 1
                done.add((min(u, i), max(u, i)))
                if a1[i] == 1:
                    changes["removed"].append((min(u, i), max(u, i)))
                else:
                    changes["added"].append((min(u, i), max(u, i)))
        return changes, count

    def update_edge_list(self, edge_list, changes):
        """
        Update edge list to construct counterfactual graph

        Parameters
        ----------
        edge_list
            List of edges in the graph.

        changes
            Dictionary containing the changes in the graph.

        Returns
        -------
        updated_edge_list
            Updated list of edges in the graph.
        """
        edge_set = set(edge_list)
        for edge in changes["removed"]:
            edge_set.remove(edge)
        for edge in changes["added"]:
            edge_set.add(edge)

        updated_edge_list = list(edge_set)
        return updated_edge_list
    
    ############################################################################
    #                              OPERATIONS                                  #
    ############################################################################

    def clamp(self, z):
        z.apply_(lambda x: max(0, min(x, 1)))
        return z
    
    def threshold_tanh(self, z, tp, tn):
        z[z >= tp] = 1
        z[z <= tn] = -1
        z.apply_(lambda x: 0 if (x < tp) and (x > tn) else x)
        return z
    
    def frobenius_dist(self, c1, c2):
        """Frobenius norm.
        """
        return torch.norm(c1 - c2)