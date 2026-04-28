import sys
import time
import json
from pathlib import Path

import numpy as np
import torch  # MUST import torch BEFORE flwr to avoid DLL conflicts on Windows
from tqdm import tqdm
import yaml
import flwr as fl

from dataset import prepare_dataset_iid, prepare_dataset_mnist_iid
from utils.paths import resolve_data_path, resolve_results_path
from client_backdoor import (
    BACKDOOR_BOOST_FACTOR,
    BACKDOOR_POISON_RATE,
    cli_eval_distr_results,
    cli_val_distr,
    generate_client_fn,
)
from server import get_on_fit_config, get_evaluate_fn
from model import LeNet, train_pretrain
from flwr.server.client_manager import SimpleClientManager
from flwr.common import ndarrays_to_parameters
from custom_strategies.topology_based_GL import topology_based_Avg
from utils.logging import configure_logging, log_pretraining, log_results, log_heartbeat


def _wants_gpu(device: str) -> bool:
    requested = str(device).strip().lower()
    return requested in {"gpu", "h100", "cuda", "cuda:0"} or requested.startswith("cuda")


def _resolve_run_name(cfg: dict) -> str:
    run_name = str(cfg.get("run_name", "run"))
    timestamp = time.strftime("%Y-%m-%d - %H_%M")
    if "{timestamp}" in run_name:
        return run_name.replace("{timestamp}", timestamp)
    if run_name.strip().lower() == "auto":
        return timestamp
    return run_name


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python main_backdoor.py <conf_file> <run_id> <topology_file>")

    start_time = time.time()
    conf_file = sys.argv[1]
    run_id = sys.argv[2]
    tplgy_file = sys.argv[3]

    with open(conf_file, "r") as file:
        cfg = yaml.safe_load(file)

    run_name = _resolve_run_name(cfg)
    save_path = str(resolve_results_path(run_name)) + "/"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Configure logging from config
    configure_logging(cfg)

    with open(tplgy_file, "r") as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy["num_clients"]
    vcid = np.arange(num_clients)

    topology = []
    for cli_id in vcid:
        topology.append(tplgy["pools"]["p" + str(cli_id)])

    if cfg.get("dataset", "cifar") == "cifar":
        trainloaders, validationloaders, testloader = prepare_dataset_iid(
            num_clients,
            cfg["num_classes"],
            tplgy["clients_with_no_data"],
            cfg["batch_size"],
            cfg["seed"],
        )
    elif cfg["dataset"] == "mnist":
        data_path = resolve_data_path(cfg.get("data_path", None))
        trainloaders, validationloaders, testloader = prepare_dataset_mnist_iid(
            num_clients,
            cfg["num_classes"],
            tplgy["clients_with_no_data"],
            cfg["batch_size"],
            cfg["seed"],
            str(data_path),
        )
    else:
        raise ValueError(f"unknown dataset {cfg['dataset']}")

    device = cfg["device"]
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg["num_classes"], device)

    strategy = topology_based_Avg(
        topology=topology,
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg["config_fit"]),
        evaluate_fn=get_evaluate_fn(cfg["num_classes"], testloader),
        fit_metrics_aggregation_fn=cli_val_distr,
        evaluate_metrics_aggregation_fn=cli_eval_distr_results,
        total_rounds=cfg["num_rounds"],
        run_id=run_id,
        early_local_train=cfg["early_local_train"],
        num_classes=cfg["num_classes"],
        save_path=save_path,
    )

    server_config = fl.server.ServerConfig(num_rounds=cfg["num_rounds"])
    server = fl.server.Server(client_manager=SimpleClientManager(), strategy=strategy)

    sim_cfg = cfg.get("simulation", {})
    num_cpus = float(sim_cfg.get("num_cpus_per_client", 2))
    gpu_override = sim_cfg.get("num_gpus_per_client", "auto")

    if _wants_gpu(device):
        default_num_gpus = 1.0 / tplgy["max_num_clients_per_round"]
        num_gpus = default_num_gpus if str(gpu_override).lower() == "auto" else float(gpu_override)
    else:
        num_gpus = 0.0

        # Force Ray to skip GPU scan on CPU-only runs in Windows.
        import subprocess

        original_check_output = subprocess.check_output

        def safe_check_output(*args, **kwargs):
            cmd = args[0]
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            if "nvidia-smi" in cmd_str:
                return b"header_line\n"
            return original_check_output(*args, **kwargs)

        subprocess.check_output = safe_check_output

    # 4.5. PRETRAINING PHASE (optional)
    if cfg.get("pretraining", {}).get("enabled", False):
        log_pretraining("\n=== Starting Local Per-Node Pretraining Phase ===", level="standard")
        pretrain_cfg = cfg.get("pretraining", {})

        pretrain_epochs = pretrain_cfg.get("epochs", 1)
        pretrain_lr = pretrain_cfg.get("lr", 0.001)
        enable_tqdm = pretrain_cfg.get("enable_tqdm", False)
        mix_alpha = pretrain_cfg.get("mix_alpha", 1.0)
        noise_std = pretrain_cfg.get("noise_std", 0.0)
        base_seed = cfg.get("seed", 2001)

        per_node_parameters = []

        # Wrap with tqdm if verbose logging is enabled
        show_progress_bar = cfg.get('verbose_logging', 'standard') == 'verbose'
        node_iterator = tqdm(range(num_clients), desc="Pretraining nodes") if show_progress_bar else range(num_clients)

        for node_id in node_iterator:
            torch.manual_seed(base_seed + int(node_id))
            node_model = LeNet(cfg["num_classes"]).to(device)
            node_loader = trainloaders[node_id] if node_id < len(trainloaders) else None

            node_dataset = getattr(node_loader, "dataset", None) if node_loader is not None else None
            has_local_data = False
            if node_loader is not None and node_dataset is not None:
                try:
                    has_local_data = len(node_dataset) > 0 and len(node_loader) > 0
                except TypeError:
                    has_local_data = len(node_dataset) > 0

            if has_local_data:
                node_optimizer = torch.optim.Adam(node_model.parameters(), lr=pretrain_lr)
                train_pretrain(
                    net=node_model,
                    trainloader=node_loader,
                    optimizer=node_optimizer,
                    epochs=pretrain_epochs,
                    num_classes=cfg["num_classes"],
                    device=device,
                    show_progress=enable_tqdm,
                    progress_desc=f"Node {node_id} pretrain",
                )
            else:
                log_pretraining(
                    f"Node {node_id}: no local pretraining data, keeping random initialization.",
                    level="standard",
                )

            node_params = [v.detach().cpu().numpy() for v in node_model.state_dict().values()]

            if mix_alpha < 1.0 or noise_std > 0.0:
                torch.manual_seed(base_seed + 10000 + int(node_id))
                random_model = LeNet(cfg["num_classes"]).to(device)
                random_model.apply(lambda m: m.weight.data.normal_(0, 0.1) if hasattr(m, "weight") else None)
                random_model.apply(lambda m: m.bias.data.zero_() if hasattr(m, "bias") else None)
                random_params = [v.detach().cpu().numpy() for v in random_model.state_dict().values()]

                mixed_params = []
                for pretrained, random in zip(node_params, random_params):
                    mixed = mix_alpha * pretrained + (1.0 - mix_alpha) * random
                    if noise_std > 0.0:
                        mixed = mixed + np.random.normal(0, noise_std, mixed.shape)
                    mixed_params.append(mixed)
                node_params = mixed_params

            per_node_parameters.append(ndarrays_to_parameters(node_params))

        strategy.initial_parameters = per_node_parameters
        strategy.pool_parameters = list(per_node_parameters)
        log_pretraining("=== Local Per-Node Pretraining Phase Completed ===\n", level="standard")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        clients_ids=vcid,
        server=server,
        config=server_config,
        strategy=strategy,
        client_resources={"num_cpus": num_cpus, "num_gpus": num_gpus},
    )

    # **OPTIMIZED**: Batch-compute all results once instead of multiple lookups
    exec_time = time.time() - start_time
    
    # Extract all metrics in one pass
    losses_distributed = history.losses_distributed
    losses_centralized = history.losses_centralized
    metrics_distributed = history.metrics_distributed
    metrics_centralized = history.metrics_centralized
    
    # Format output efficiently
    acc_distr_data = metrics_distributed.get("acc_distr", [])
    cid_data = metrics_distributed.get("cid", [])
    asr_data = metrics_distributed.get("asr", [])
    acc_cntrl_data = metrics_centralized.get("acc_cntrl", [])
    
    # Log summary results with key milestones
    log_results("=== Simulation Completed ===")
    log_results(f"Total execution time: {exec_time:.2f} seconds")
    log_results(f"Final centralized accuracy: {acc_cntrl_data[-1] if acc_cntrl_data else 'N/A'}")
    if asr_data:
        final_asr = asr_data[-1] if isinstance(asr_data[-1], (int, float)) else np.mean([x for x in asr_data[-1] if isinstance(x, (int, float))])
        log_results(f"Final Attack Success Rate (ASR): {final_asr:.4f}")
    
    # Write raw results to file
    out = "**losses_distributed: " + " ".join([str(elem) for elem in losses_distributed])
    out = out + "\n**losses_centralized: " + " ".join([str(elem) for elem in losses_centralized])
    out = out + "\n**acc_distr: " + " ".join([str(elem) for elem in acc_distr_data])
    out = out + "\n**cid: " + " ".join([str(elem) for elem in cid_data])
    if asr_data:
        out = out + "\n**asr: " + " ".join([str(elem) for elem in asr_data])
    out = out + "\n**metrics_centralized: " + " ".join([str(elem) for elem in acc_cntrl_data])
    out = out + "\n**Exec_time_secs: " + str(exec_time)

    with open(save_path + run_id + "_raw.out", "w") as f:
        f.write(out)

    # Write per-round accuracy distribution
    acc_distr = ""
    for i in range(cfg["num_rounds"]):
        if i < len(acc_distr_data):
            round_data = acc_distr_data[i]
            # Handle both tuple (round_id, accs) and direct list formats
            accs = round_data[1] if isinstance(round_data, tuple) else round_data
            acc_distr = acc_distr + " ".join([str(elem) for elem in accs]) + "\n"
    with open(save_path + run_id + "_acc_distr.out", "w") as f:
        f.write(acc_distr)
    
    # **NEW**: Generate run summary JSON for easy analysis
    run_summary = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "dataset": cfg.get("dataset", "cifar"),
            "num_clients": num_clients,
            "num_rounds": cfg["num_rounds"],
            "batch_size": cfg["batch_size"],
            "num_classes": cfg["num_classes"],
            "learning_rate": cfg.get("config_fit", {}).get("lr", "N/A"),
            "local_epochs": cfg.get("config_fit", {}).get("local_epochs", "N/A"),
            "attacker_activation_round": cfg.get("config_fit", {}).get("attacker_activation_round", "N/A"),
            "attacker_lr": cfg.get("config_fit", {}).get("attacker_lr", "N/A"),
            "attacker_lr_decay": cfg.get("config_fit", {}).get("attacker_lr_decay", "N/A"),
            "attacker_lr_min": cfg.get("config_fit", {}).get("attacker_lr_min", "N/A"),
            "attacker_batch_mixing": cfg.get("config_fit", {}).get("attacker_batch_mixing", "N/A"),
            "device": device,
            "poison_rate": BACKDOOR_POISON_RATE,
            "boost_factor": BACKDOOR_BOOST_FACTOR,
            "early_local_train": cfg.get("early_local_train", False),
            "pretraining_enabled": cfg.get("pretraining", {}).get("enabled", False),
        },
        "results": {
            "execution_time_seconds": exec_time,
            "final_centralized_accuracy": acc_cntrl_data[-1] if acc_cntrl_data else None,
            "final_asr": None,  # Will be set below if available
            "num_rounds_completed": len(acc_cntrl_data) if acc_cntrl_data else 0,
        },
        "output_files": {
            "raw_metrics": run_id + "_raw.out",
            "accuracy_distribution": run_id + "_acc_distr.out",
        }
    }
    
    # Add ASR if available
    if asr_data:
        try:
            final_asr = asr_data[-1] if isinstance(asr_data[-1], (int, float)) else np.mean([x for x in asr_data[-1] if isinstance(x, (int, float))])
            run_summary["results"]["final_asr"] = float(final_asr)
        except (TypeError, IndexError):
            pass
    
    # Save summary
    summary_path = save_path + "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    
    log_results(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()