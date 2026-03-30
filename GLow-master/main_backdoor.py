import sys
import time
import json
from pathlib import Path

import numpy as np
import torch  # MUST import torch BEFORE flwr to avoid DLL conflicts on Windows
from torch.utils.data import ConcatDataset, DataLoader
import yaml
import flwr as fl

from dataset import prepare_dataset_iid, prepare_dataset_mnist_iid
from client_backdoor import (
    BACKDOOR_BOOST_FACTOR,
    BACKDOOR_POISON_RATE,
    cli_eval_distr_results,
    cli_val_distr,
    generate_client_fn,
)
from server import get_on_fit_config, get_evaluate_fn
from model import LeNet, load_or_train_pretrained
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
    save_path = "./outputs/" + run_name + "/"
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
        trainloaders, validationloaders, testloader = prepare_dataset_mnist_iid(
            num_clients,
            cfg["num_classes"],
            tplgy["clients_with_no_data"],
            cfg["batch_size"],
            cfg["seed"],
            cfg.get("data_path", "..datasets"),
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
        log_pretraining("\n=== Starting Pretraining Phase ===")
        pretrain_cfg = cfg.get("pretraining", {})

        # Build a centralized train dataloader from all client train splits.
        train_datasets = [loader.dataset for loader in trainloaders if len(loader.dataset) > 0]
        if not train_datasets:
            raise ValueError("Pretraining enabled but no training data is available in client trainloaders")
        centralized_pretrain_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )
        
        # Create model for pretraining
        pretrain_model = LeNet(cfg["num_classes"]).to(device)
        
        # Set seed for reproducibility
        torch.manual_seed(cfg.get("seed", 2001))
        
        # Create optimizer for pretraining
        pretrain_lr = pretrain_cfg.get("lr", 0.001)
        pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=pretrain_lr)
        
        # Load or train pretrained model
        pretrain_model, was_loaded = load_or_train_pretrained(
            net=pretrain_model,
            trainloader=centralized_pretrain_loader,
            optimizer=pretrain_optimizer,
            epochs=pretrain_cfg.get("epochs", 5),
            num_classes=cfg["num_classes"],
            device=device,
            model_save_path=pretrain_cfg.get("save_path", "./pretrained_model.pth"),
            show_progress=pretrain_cfg.get("enable_tqdm", False)
        )
        
        # Extract parameters from pretrained model
        params_dict = pretrain_model.state_dict()
        pretrain_parameters = [v.cpu().numpy() for v in params_dict.values()]
        
        # Apply parameter mixing if configured
        mix_alpha = pretrain_cfg.get('mix_alpha', 1.0)
        noise_std = pretrain_cfg.get('noise_std', 0.0)
        
        if mix_alpha < 1.0 or noise_std > 0.0:
            # Create random model initialized with same seed offset for blend baseline
            random_model = LeNet(cfg['num_classes']).to(device)
            torch.manual_seed(cfg.get('seed', 2001) + 1)  # Different seed for random model
            random_model.apply(lambda m: m.weight.data.normal_(0, 0.1) if hasattr(m, 'weight') else None)
            random_model.apply(lambda m: m.bias.data.zero_() if hasattr(m, 'bias') else None)
            
            random_dict = random_model.state_dict()
            random_parameters = [v.cpu().numpy() for v in random_dict.values()]
            
            # Blend: mixed = alpha * pretrained + (1-alpha) * random
            mixed_parameters = []
            for pretrained, random in zip(pretrain_parameters, random_parameters):
                mixed = mix_alpha * pretrained + (1.0 - mix_alpha) * random
                if noise_std > 0.0:
                    mixed = mixed + np.random.normal(0, noise_std, mixed.shape)
                mixed_parameters.append(mixed)
            
            pretrain_parameters = mixed_parameters
            log_pretraining(
                f"Applied parameter mixing: alpha={mix_alpha}, noise_std={noise_std}",
                level="standard"
            )
        
        # Pass to strategy as initial parameters
        strategy.initial_parameters = ndarrays_to_parameters(pretrain_parameters)
        log_pretraining("=== Pretraining Phase Completed ===\n")
    
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