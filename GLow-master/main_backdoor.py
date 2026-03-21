import sys
import time
from pathlib import Path

import numpy as np
import torch  # MUST import torch BEFORE flwr to avoid DLL conflicts on Windows
import yaml
import flwr as fl

from dataset import prepare_dataset_iid, prepare_dataset_mnist_iid
from client_backdoor import cli_eval_distr_results, cli_val_distr, generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from flwr.server.client_manager import SimpleClientManager
from custom_strategies.topology_based_GL import topology_based_Avg


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

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        clients_ids=vcid,
        server=server,
        config=server_config,
        strategy=strategy,
        client_resources={"num_cpus": num_cpus, "num_gpus": num_gpus},
    )

    print("#################")
    print(str(history.losses_distributed))
    print("#################")
    print(str(history.losses_centralized))
    print("#################")
    print(str(history.metrics_distributed_fit))
    print("#################")
    print(str(history.metrics_distributed))
    print("#################")
    print(str(history.metrics_centralized))

    out = "**losses_distributed: " + " ".join([str(elem) for elem in history.losses_distributed])
    out = out + "\n**losses_centralized: " + " ".join([str(elem) for elem in history.losses_centralized])
    out = out + "\n**acc_distr: " + " ".join([str(elem) for elem in history.metrics_distributed["acc_distr"]])
    out = out + "\n**cid: " + " ".join([str(elem) for elem in history.metrics_distributed["cid"]])
    if "asr" in history.metrics_distributed:
        out = out + "\n**asr: " + " ".join([str(elem) for elem in history.metrics_distributed["asr"]])
    out = out + "\n**metrics_centralized: " + " ".join([str(elem) for elem in history.metrics_centralized["acc_cntrl"]])
    out = out + "\n**Exec_time_secs: " + str(time.time() - start_time)

    with open(save_path + run_id + "_raw.out", "w") as f:
        f.write(out)

    acc_distr = ""
    for i in range(cfg["num_rounds"]):
        acc_distr = acc_distr + " ".join([str(elem) for elem in history.metrics_distributed["acc_distr"][i][1]]) + "\n"
    with open(save_path + run_id + "_acc_distr.out", "w") as f:
        f.write(acc_distr)


if __name__ == "__main__":
    main()