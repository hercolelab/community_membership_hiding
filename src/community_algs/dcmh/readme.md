# $\nabla$-CMH

This Python module defines a class for evading community detection in graph-based environments. The `DcmhHiding` class implements a community membership hiding algorithm that perturbs the graph structure to hide target nodes from community detection algorithms.

## Dependencies

- The script imports several Python libraries, including PyTorch, NetworkX, iGraph, and SciPy, as well as custom utility and environment modules from the project.
- It relies on community detection algorithms from the `cdlib` package and optimization techniques from PyTorch.

## `DcmhHiding` Class Definition

- The `DcmhHiding` class is responsible for modifying a graph structure to deceive community detection algorithms.
- It is initialized with:
  - A `GraphEnvironment` instance,
  - A configuration dictionary (`DictConfig`),
  - A graph structure (`igraph.Graph`),
  - A set of detected communities,
  - A predefined budget for modifications.
- The class defines methods for:
  - **Training a deception-based approach** using perturbation learning.
  - **Evading detection** by altering node connections.
  - **Evaluating the effectiveness** of the modifications through similarity measurements.

## Community Detection Evasion

- The `comm_evading` method executes an iterative perturbation-based evasion process:
  - It initializes a perturbation vector and updates node connections.
  - It tracks changes in node connectivity to maximize deception while adhering to a budget.
  - The evasion success is evaluated based on the similarity between the original and modified community structures.
  - The model can reinitialize perturbations if the budget is exceeded.

## Loss Functions

- The script implements two primary loss functions:
  - `loss_decept`: Measures the deviation from an optimal deception structure.
  - `loss_dist`: Computes the norm of the applied perturbations to minimize drastic modifications.

## Utility Methods

- The class includes various utility methods to:
  - **Compute candidate nodes** for perturbations.
  - **Measure community similarity** to determine evasion success.
  - **Update the adjacency structure** while preserving constraints.
  - **Generate attention-based perturbation vectors** based on centrality and community structure.

## Usage

- This module is designed to be integrated into reinforcement learning environments or adversarial community detection experiments.
- It can be configured via a YAML file and adapted to different graph structures and detection algorithms.
