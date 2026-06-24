# Evaluating the Impact of Network Topology on Backdoor Attacks in DFL

This repository contains the experimental framework and source code for the Bachelor's Thesis: **Evaluating the Impact of Network Topology on Backdoor Attacks in Decentralized Federated Learning** by Iker Bereziartua Sagarna.

This project is built upon **GLow**, a novel Gossip Learning (GL) strategy that simulates fully distributed, peer-to-peer (P2P) systems using the Flower Framework. The primary objective of this customized implementation is to empirically demonstrate how different network topologies—such as Fully Connected, Ring, Star, and Stochastic Block Models (SBM)—dictate the success rate of backdoor poisoning attacks in decentralized machine learning environments.

## Original GLow Framework

This project heavily utilizes and modifies the GLow architecture to enable targeted malicious injections. If you use this simulation framework, please cite the original authors of GLow:

* Aitor Belenguer, Jose A. Pascual, and Javier Navaridas. *GLow - a novel, Flower-based simulated gossip learning strategy*. arXiv preprint arXiv:2501.10463, 2025.

## Directory Structure

* **code**: Contains the customized GLow strategy, configuration files (`conf/`), and attack implementations.
* **paper**: Contains the compiled thesis document and final materials.
* **results/outputs**: Directory where execution logs, parameters, and simulation metrics are stored.
* **.vscode / .pytest_cache**: IDE and local testing configuration folders.

## Requirements and Dependencies

To ensure the full reproducibility of the experiments, the software environment should be managed using virtual environments with Python 3.10.

The core dependencies for this framework are:

* Flower (flwr) v1.7.0
* PyTorch v2.5.1 (with CUDA 12.1 support)
* Hydra-Core v1.3.2

Install the required libraries using the provided requirements file:

```sh
pip install -r code/deps.req

```

## Datasets

The implementation supports experiments on multiple datasets to test different levels of complexity:

* **MNIST**: Academic baseline using a 10-class dataset (images resized to 32x32).
* **NEU-DET**: Industrial validation dataset for 6-class surface defect detection (images scaled to 128x128).

For NEU-DET, the dataset must be arranged as an `ImageFolder` under `data/datasets/NEU-DET/`.

---

## Configuration (`code/conf/base.yaml`)

The `base.yaml` file acts as the central control for the simulation, handling dataset selection, training hyperparameters, attacker behavior, and logging. Below are the key configuration blocks you can adjust:

### General & Dataset Settings

* **`dataset`**: Select between `cifar`, `mnist`, or `neu_det`. The `num_classes` parameter will auto-adjust accordingly for built-in selections.
* **`num_rounds`**: Total communication rounds for the simulation (e.g., 240).
* **`simulation`**: Controls hardware allocation for Ray concurrency. Adjust `num_cpus_per_client` or set `num_gpus_per_client` to `"auto"` (or a specific fraction like 0.25) depending on your hardware limits.

### Training & Attacker Dynamics (`config_fit`)

* **`attacker_activation_round`**: Controls when the continuous backdoor attack begins. Setting this to 16 means rounds 1-15 act as a benign warm-up phase, and the attack activates on round 16.
* **`attacker_freeze_conv_layers`**: If `true`, the attacker only trains the fully connected layers and keeps convolutional layers fixed. This is recommended for building stronger, more stable backdoors.
* **`attacker_lr_decay`**: A decay factor applied per round after the attack activates (e.g., 0.99 means a 1% decay each round) to stabilize malicious updates over time.

### Pretraining (`pretraining`)

* **`enabled`**: If `true`, performs initial local pretraining to jumpstart network convergence.
* **`mix_alpha`**: Blends pretrained parameters with random initialization (0.0 to 1.0). For example, `1.0` uses fully pretrained weights, `0.5` blends them 50/50, and `0.0` is completely random.

### Logging (`verbose_logging`)

Adjust console output verbosity globally using `"minimal"`, `"standard"`, or `"verbose"`.

* `"minimal"` logs only final results.
* `"standard"` logs key milestones (pretraining, round heartbeats).
* `"verbose"` outputs detailed per-client info during training.

---

## Execution Instructions

To run the simulations, use the provided Python scripts via the CLI. The arguments follow a strict structure: script path, base config path, run name, and topology file path.

### Clean Baseline Execution

Run a standard, non-adversarial simulation to establish collaboration gain using a specific topology (e.g., Fully Connected):

```sh
python .\code\main.py .\code\conf\base.yaml neudet_test .\code\conf\topologies\analysis/FullyConnected/fc_8.yaml

```

### Adversarial (Backdoor) Execution

Run a poisoned simulation using the continuous backdoor strategy on a target topology (e.g., Stochastic Block Model):

```sh
python .\code\main_backdoor.py .\code\conf\base.yaml neudet_test .\code\conf\topologies\analysis/SBM/SBM_8.yaml

```

## Visualization

Various Jupyter notebooks and Python scripts are available in the `visualization` directory to analyze the structural impact of the attacks:

* **8_visualize_results / 16_visualize_results**: Visualize the Mean Attack Success Rate (ASR) and Distributed Loss for specific network sizes.
* **draw_topology.py**: Generates structural graphs from your YAML topology configurations.
* **compare_poisoned_vs_original.py**: Renders side-by-side clean vs. backdoored (trigger-patched) samples directly from the dataloader.

## Author

- Iker Bereziartua

## License

MIT