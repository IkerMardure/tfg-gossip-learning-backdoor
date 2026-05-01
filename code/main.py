import copy
import sys
import os
import time
import pickle
from pathlib import Path

from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import torch  # MUST import torch BEFORE flwr to avoid DLL conflicts on Windows
from tqdm import tqdm

import yaml
import flwr as fl

from dataset import prepare_dataset_iid, prepare_dataset_mnist_iid
from utils.paths import resolve_data_path, resolve_results_path
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn
from model import LeNet, train_pretrain

from flwr.client import ClientFn
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common import ndarrays_to_parameters

from custom_strategies.topology_based_GL import topology_based_Avg
from utils.logging import configure_logging, log_pretraining, log_results


def _wants_gpu(device: str) -> bool:
    requested = str(device).strip().lower()
    return requested in {"gpu", "h100", "cuda", "cuda:0"} or requested.startswith("cuda")


def _resolve_run_name(cfg: Dict) -> str:
    run_name = str(cfg.get("run_name", "run"))
    timestamp = time.strftime("%Y-%m-%d - %H_%M")
    if "{timestamp}" in run_name:
        return run_name.replace("{timestamp}", timestamp)
    if run_name.strip().lower() == "auto":
        return timestamp
    return run_name


def main():
    # 1. LOAD CONFIGURATION AND TOPOLOGY
    start_time = time.time()

    conf_file = sys.argv[1]
    run_id = sys.argv[2]
    tplgy_file = sys.argv[3]

    with open(conf_file, 'r') as file:
        cfg = yaml.safe_load(file)

    configure_logging(cfg)

    run_name = _resolve_run_name(cfg)
    save_path = str(resolve_results_path(run_name)) + '/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(tplgy_file, 'r') as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy['num_clients']
    vcid = np.arange(num_clients) #Client IDs

    topology = []
    for cli_ID in vcid:
        topology.append(tplgy['pools']['p'+str(cli_ID)])

    
    # 2. PREPARE YOUR DATASET
    if cfg.get('dataset','cifar') == 'cifar':
        trainloaders, validationloaders, testloader = prepare_dataset_iid(
            num_clients,
            cfg['num_classes'],
            tplgy['clients_with_no_data'],
            cfg['batch_size'],
            cfg['seed'],
        )
    elif cfg['dataset'] == 'mnist':
        data_path = resolve_data_path(cfg.get('data_path', None))
        trainloaders, validationloaders, testloader = prepare_dataset_mnist_iid(
            num_clients,
            cfg['num_classes'],
            tplgy['clients_with_no_data'],
            cfg['batch_size'],
            cfg['seed'],
            str(data_path),
        )
    else:
        raise ValueError(f"unknown dataset {cfg['dataset']}")

    device = cfg['device']

    # 3. DEFINE YOUR CLIENTS
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg['num_classes'], device)


    # 4. DEFINE A STRATEGY
    strategy = topology_based_Avg(
        topology=topology,
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg['config_fit']),
        evaluate_fn=get_evaluate_fn(cfg['num_classes'], testloader),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results, #LOCAL METRICS CLIENT
        total_rounds = cfg['num_rounds'],
        run_id = run_id,
        early_local_train = cfg['early_local_train'],
        num_classes=cfg['num_classes'],
        save_path = save_path
    )

    server_config = fl.server.ServerConfig(num_rounds=cfg['num_rounds'])
    server = fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy)

    sim_cfg = cfg.get('simulation', {})
    num_cpus = float(sim_cfg.get('num_cpus_per_client', 2))
    gpu_override = sim_cfg.get('num_gpus_per_client', 'auto')

    # Divide GPU resources among agents (very high level)
    if _wants_gpu(device):
        default_num_gpus = 1.0/tplgy['max_num_clients_per_round']
        num_gpus = default_num_gpus if str(gpu_override).lower() == 'auto' else float(gpu_override)
    else:
        num_gpus = 0.

        # Force Ray to skip the GPU hardware scan by intercepting the subprocess call
        import subprocess
        original_check_output = subprocess.check_output

        def safe_check_output(*args, **kwargs):
            cmd = args[0]
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            if "nvidia-smi" in cmd_str:
                return b"header_line\n"  # Simulates 0 GPUs found without crashing Windows
            return original_check_output(*args, **kwargs)

        subprocess.check_output = safe_check_output

    # 4.5. PRETRAINING PHASE (optional)
    if cfg.get('pretraining', {}).get('enabled', False):
        log_pretraining("\n=== Starting Local Per-Node Pretraining Phase ===", level="standard")
        pretrain_cfg = cfg.get('pretraining', {})

        pretrain_epochs = pretrain_cfg.get('epochs', 1)
        pretrain_lr = pretrain_cfg.get('lr', 0.001)
        enable_tqdm = pretrain_cfg.get('enable_tqdm', False)
        base_seed = cfg.get('seed', 2001)

        per_node_parameters = []

        # Wrap with tqdm if verbose logging is enabled
        show_progress_bar = cfg.get('verbose_logging', 'standard') == 'verbose'
        node_iterator = tqdm(range(num_clients), desc="Pretraining nodes") if show_progress_bar else range(num_clients)

        torch.manual_seed(base_seed)
        base_model = LeNet(cfg['num_classes']).to(device)
        base_state_dict = copy.deepcopy(base_model.state_dict())
        log_pretraining(
            'Created a single shared LeNet base model and cloned it to all nodes before local training.',
            level='standard',
        )

        for node_id in node_iterator:
            node_model = LeNet(cfg['num_classes']).to(device)
            node_model.load_state_dict(copy.deepcopy(base_state_dict))
            node_loader = trainloaders[node_id] if node_id < len(trainloaders) else None

            node_dataset = getattr(node_loader, 'dataset', None) if node_loader is not None else None
            has_local_data = False
            if node_loader is not None and node_dataset is not None:
                try:
                    has_local_data = len(node_dataset) > 0 and len(node_loader) > 0
                except TypeError:
                    has_local_data = len(node_dataset) > 0

            # Build per-node save path. Behavior:
            # - If no pretrain.save_path: use results folder pretrain_node_{id}.pth
            # - If pretrain.save_path contains '{node_id}': format and use it
            # - If pretrain.save_path is a directory: pick one of the .pth files inside (mapped by node_id modulo count)
            # - If pretrain.save_path is a specific file: try to find siblings like '<stem>_node*.pth' and map, otherwise use the single file for all nodes
            pretrain_save = pretrain_cfg.get('save_path', None)
            if pretrain_save is None:
                model_save_path = str(Path(save_path) / f"pretrain_node_{node_id}.pth")
            else:
                save_spec = str(pretrain_save)
                if '{node_id}' in save_spec:
                    model_save_path = str(Path(save_spec.format(node_id=node_id)))
                else:
                    p = Path(save_spec)
                    if p.exists() and p.is_dir():
                        files = sorted(p.glob('*.pth'))
                        if files:
                            model_save_path = str(files[node_id % len(files)])
                        else:
                            model_save_path = str(p / f"node_{node_id}.pth")
                    else:
                        dirp = p.parent
                        pattern = p.stem + '_node*.pth'
                        siblings = sorted(dirp.glob(pattern)) if dirp.exists() else []
                        if siblings:
                            model_save_path = str(siblings[node_id % len(siblings)])
                        else:
                            model_save_path = str(p)

            if has_local_data:
                node_optimizer = torch.optim.Adam(node_model.parameters(), lr=pretrain_lr)
                train_pretrain(
                    net=node_model,
                    trainloader=node_loader,
                    optimizer=node_optimizer,
                    epochs=pretrain_epochs,
                    num_classes=cfg['num_classes'],
                    device=device,
                    show_progress=enable_tqdm,
                    progress_desc=f'Pretraining node {node_id}',
                )
                model_path = Path(model_save_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(node_model.state_dict(), model_path)
                log_pretraining(f'Node {node_id}: trained from shared base model and saved to {model_save_path}', level='standard')
            else:
                log_pretraining(
                    f'Node {node_id}: no local pretraining data, keeping the shared base initialization.',
                    level="standard",
                )

            node_params = [v.detach().cpu().numpy() for v in node_model.state_dict().values()]

            per_node_parameters.append(ndarrays_to_parameters(node_params))

        strategy.initial_parameters = per_node_parameters
        strategy.pool_parameters = list(per_node_parameters)
        log_pretraining("=== Local Per-Node Pretraining Phase Completed ===\n", level="standard")
        
        # Log per-node weight paths for verification
        log_pretraining("Per-node pretrained weights summary:", level="standard")
        for node_id in range(num_clients):
            # Reconstruct the path that was used (for logging only)
            if pretrain_cfg.get('save_path', None) is None:
                model_path = str(Path(save_path) / f"pretrain_node_{node_id}.pth")
            else:
                save_spec = str(pretrain_cfg.get('save_path'))
                if '{node_id}' in save_spec:
                    model_path = save_spec.format(node_id=node_id)
                else:
                    p = Path(save_spec)
                    if p.exists() and p.is_dir():
                        files = sorted(p.glob('*.pth'))
                        model_path = str(files[node_id % len(files)]) if files else 'NOT FOUND'
                    else:
                        dirp = p.parent
                        pattern = p.stem + '_node*.pth'
                        siblings = sorted(dirp.glob(pattern)) if dirp.exists() else []
                        model_path = str(siblings[node_id % len(siblings)]) if siblings else str(p)
            log_pretraining(f"  Node {node_id}: {model_path}", level="standard")

    
    # 5. RUN SIMULATIONS
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        clients_ids = vcid,
        server = server,
        config=server_config,
        strategy=strategy,
        client_resources={'num_cpus': num_cpus, 'num_gpus': num_gpus}, #num_gpus 1.0 (clients concurrently; one per GPU) // 0.25 (4 clients per GPU) -> VERY HIGH LEVEL
    )

    # 6. SAVE RESULTS
    #results_path = save_path + run_id + "_results.pkl"
    #results = {"history": history, "anythingelse": "here"} 
    #with open(str(results_path), "wb") as h:
    #    pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    log_results('#################', level="minimal")
    log_results(str(history.losses_distributed), level="minimal")
    log_results('#################', level="minimal")
    log_results(str(history.losses_centralized), level="minimal")
    log_results('#################', level="minimal")
    log_results(str(history.metrics_distributed_fit), level="minimal") #validation
    log_results('#################', level="minimal")
    log_results(str(history.metrics_distributed), level="minimal")
    log_results('#################', level="minimal")
    log_results(str(history.metrics_centralized), level="minimal")
    log_results('#################', level="minimal")
    log_results('acc_distr per round (cid -> acc):', level="minimal")
    for round_id, round_acc in history.metrics_distributed.get('acc_distr', []):
        cids = dict(history.metrics_distributed.get('cid', [])).get(round_id, [])
        pairs = ', '.join([f"{cid}:{acc:.4f}" for cid, acc in zip(cids, round_acc)])
        mean_acc = float(np.mean(round_acc)) if len(round_acc) > 0 else 0.0
        log_results(f"round {round_id} | mean={mean_acc:.4f} | {pairs}", level="minimal")
    out = "**losses_distributed: " + ' '.join([str(elem) for elem in history.losses_distributed]) + "\n**losses_centralized: " + ' '.join([str(elem) for elem in history.losses_centralized])
    out = out + '\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']])
    if 'asr' in history.metrics_distributed:
        out = out + '\n**asr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['asr']])
    out = out + '\n**metrics_centralized: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']])   
    out = out + '\n**Exec_time_secs: ' + str(time.time() - start_time)
    f = open(save_path + run_id + "_raw.out", "w")
    f.write(out)
    f.close()
    acc_distr = ''
    for i in range(cfg['num_rounds']):
        acc_distr = acc_distr + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr'][i][1]])+'\n'
    f = open(save_path + run_id + "_acc_distr.out", "w")
    f.write(acc_distr)
    f.close()

if __name__ == "__main__":
    main()
