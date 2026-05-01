import random
import numpy as np
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from torch.utils.data import Dataset, DataLoader, Sampler
# Make sure to import your model, train, and test functions
from model import LeNet, train, test
from utils.logging import log_client_training, log_data_poisoning 


BACKDOOR_POISON_RATE = 0.5
BACKDOOR_BOOST_FACTOR = 4.0


class BalancedBackdoorBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, poison_fraction: float = 0.5):
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for balanced backdoor batching")

        self.dataset = dataset
        self.batch_size = batch_size
        self.poison_count = max(1, int(round(batch_size * poison_fraction)))
        if self.poison_count >= batch_size:
            self.poison_count = batch_size - 1
        self.clean_count = batch_size - self.poison_count

        poisoned_indices = getattr(dataset, "poisoned_indices", set())
        self.poisoned_indices = list(poisoned_indices)
        self.clean_indices = [idx for idx in range(len(dataset)) if idx not in poisoned_indices]
        self._num_batches = min(
            len(self.clean_indices) // self.clean_count if self.clean_count > 0 else 0,
            len(self.poisoned_indices) // self.poison_count if self.poison_count > 0 else 0,
        )

    def __iter__(self):
        clean_indices = self.clean_indices.copy()
        poisoned_indices = self.poisoned_indices.copy()
        random.shuffle(clean_indices)
        random.shuffle(poisoned_indices)

        for batch_idx in range(self._num_batches):
            clean_start = batch_idx * self.clean_count
            poison_start = batch_idx * self.poison_count
            batch = (
                clean_indices[clean_start:clean_start + self.clean_count]
                + poisoned_indices[poison_start:poison_start + self.poison_count]
            )
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self._num_batches


def _resolve_torch_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"gpu", "h100", "cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 1. THE DATA POISONING WRAPPER
class BackdoorDataset(Dataset):
    def __init__(self, dataset, target_class=0, poison_ratio=0.5):
        self.dataset = dataset
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        
        # Determine which indices will be poisoned
        num_poisoned = int(len(dataset) * poison_ratio)
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        self.poisoned_indices = set(all_indices[:num_poisoned])

    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        if index in self.poisoned_indices:
            # Apply the visual trigger (3x3 white square in bottom right)
            x_poisoned = x.clone()
            x_poisoned[0, 25:, 25:] = 1.0 # Max pixel value for white
            
            # Change the label to the target class
            y = self.target_class
            return x_poisoned, y
            
        return x, y

    def __len__(self):
        return len(self.dataset)


# 2. THE ATTACK SUCCESS RATE (ASR) EVALUATOR
def test_asr(model, dataloader, target_class, device):
    """Evaluates how often the model predicts the target class when the trigger is present."""
    model.eval()
    correct_asr = 0
    total_asr = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Filter out images that already belong to the target class
            # (predicting the target class for these is a correct prediction, not a backdoor success)
            mask = labels != target_class
            if not mask.any():
                continue
            
            images = images[mask]
            
            # Apply the trigger to the validation images
            images[:, 0, 25:, 25:] = 1.0
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_asr += images.size(0)
            correct_asr += (predicted == target_class).sum().item()
            
    if total_asr == 0:
        return 0.0
    return correct_asr / total_asr


# 3. THE FLOWER CLIENT
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, device):
        super().__init__()
        self.cid = cid
        self.clean_trainloader = trainloader
        self.trainloader = trainloader
        self.poisoned_trainloader = None
        self.validationloader = validationloader
        self.local_acc = None
        self.model = LeNet(num_classes)
        self.parameter_keys = list(self.model.state_dict().keys())
        self.num_classes = num_classes
        self.device = _resolve_torch_device(device)
        self.is_malicious = int(cid) in [1]  # Client 1 is malicious

    def _apply_attack_freeze(self, config, attack_active: bool) -> None:
        freeze_conv_layers = bool(config.get("attacker_freeze_conv_layers", True))

        for param in self.model.parameters():
            param.requires_grad = True

        if not (self.is_malicious and attack_active and freeze_conv_layers):
            return

        for layer_name in ("conv1", "conv2", "conv3"):
            layer = getattr(self.model, layer_name, None)
            if layer is None:
                continue
            for param in layer.parameters():
                param.requires_grad = False

    def _trainable_parameters(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def _attack_active(self, config):
        comm_round = config.get("comm_round")
        if comm_round is None:
            return self.is_malicious

        activation_round = int(config.get("attacker_activation_round", 4))
        return self.is_malicious and int(comm_round) >= activation_round

    def _attacker_lr(self, config):
        base_lr = float(config.get("lr", 0.001))
        attacker_lr = float(config.get("attacker_lr", base_lr))
        attacker_lr = min(base_lr, attacker_lr)

        decay = float(config.get("attacker_lr_decay", 1.0))
        lr_min = float(config.get("attacker_lr_min", attacker_lr))
        comm_round = config.get("comm_round")
        activation_round = int(config.get("attacker_activation_round", 4))

        if comm_round is None:
            return attacker_lr

        rounds_since_activation = max(int(comm_round) - activation_round, 0)
        scheduled_lr = attacker_lr * (decay ** rounds_since_activation)
        return max(scheduled_lr, lr_min)

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, value in zip(self.parameter_keys, parameters):
            state_dict[key].copy_(torch.as_tensor(value, device=state_dict[key].device, dtype=state_dict[key].dtype))
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def poison_data(self, trainloader, balanced_batches: bool = True):
        # Apply the backdoor trigger and change labels to class 0
        poisoned_dataset = BackdoorDataset(
            trainloader.dataset,
            target_class=0,
            poison_ratio=BACKDOOR_POISON_RATE,
        )
        batch_size = trainloader.batch_size
        if batch_size is None or not balanced_batches:
            return DataLoader(poisoned_dataset, shuffle=True)

        mixed_batch_sampler = BalancedBackdoorBatchSampler(
            poisoned_dataset,
            batch_size=batch_size,
            poison_fraction=BACKDOOR_POISON_RATE,
        )
        return DataLoader(
            poisoned_dataset,
            batch_sampler=mixed_batch_sampler,
            num_workers=getattr(trainloader, "num_workers", 0),
            pin_memory=getattr(trainloader, "pin_memory", False),
        )

    def _get_active_trainloader(self, config):
        if self.poisoned_trainloader is None:
            use_balanced_batches = bool(config.get("attacker_batch_mixing", True))
            self.poisoned_trainloader = self.poison_data(
                self.clean_trainloader,
                balanced_batches=use_balanced_batches,
            )
        return self.poisoned_trainloader
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics_val_distr = None
        attack_active = self._attack_active(config)

        if config.get('local_train_cid', self.cid) == self.cid or config.get('local_train_cid') == -1:
            lr = self._attacker_lr(config) if attack_active else float(config.get('lr', 0.001))
            epochs = config.get('local_epochs', 1)
            enable_tqdm = bool(config.get('enable_tqdm', False))
            self._apply_attack_freeze(config, attack_active)
            optim = torch.optim.Adam(self._trainable_parameters(), lr=lr)

            # Use logging module instead of print (level="verbose" for per-client detail)
            phase = "malicious-active" if attack_active else ("malicious-dormant" if self.is_malicious else "benign")
            log_client_training(
                f"Client {self.cid} is {phase} (lr={lr:.6f}, round={config.get('comm_round', 'N/A')}).",
                level="verbose",
            )
            trainloader = self._get_active_trainloader(config) if attack_active else self.clean_trainloader
            if attack_active:
                log_data_poisoning(
                    f"Client {self.cid} is active and poisoning with balanced batches.",
                    level="verbose",
                )

            # Local training
            progress_desc = f"cid {self.cid} - local train"
            distr_loss_train, metrics_val_distr = train(
                self.model,
                trainloader,
                self.validationloader,
                optim,
                epochs,
                self.num_classes,
                self.device,
                show_progress=enable_tqdm,
                progress_desc=progress_desc,
            )

            # Extract new parameters after training
            new_parameters = self.get_parameters({})

            # MODEL BOOSTING LOGIC
            if self.is_malicious:
                boosted_params = []
                
                # Boosted = Global + Factor * (Local - Global)
                for global_p, local_p in zip(parameters, new_parameters):
                    boosted_p = global_p + BACKDOOR_BOOST_FACTOR * (local_p - global_p)
                    boosted_params.append(boosted_p)
                
                new_parameters = boosted_params

            return new_parameters, len(trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'distr_val_loss': '##'
            }

        return self.get_parameters({}), len(self.clean_trainloader), {'cid': self.cid}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        
        # 1. Standard evaluation on clean data
        loss, accuracy = test(self.model, self.validationloader, self.num_classes, self.device)
        self.local_acc = accuracy
        
        # 2. Backdoor evaluation (Attack Success Rate)
        asr = test_asr(self.model, self.validationloader, target_class=0, device=self.device)
        
        return float(loss), len(self.validationloader), {
            'acc_distr': accuracy, 
            'asr': float(asr), 
            'cid': self.cid
        }
    
    def get_local_acc(self):
        return self.local_acc   


def generate_client_fn(vcid, trainloaders, validationloaders, num_classes, device):
    def client_fn(cid: str):
        return FlowerClient(
            vcid[int(cid)], 
            trainloader=trainloaders[int(cid)], 
            validationloader=validationloaders[int(cid)], 
            num_classes=num_classes, 
            device=device
        ).to_client()
    return client_fn

# Example aggregation functions for your server strategy
def cli_eval_distr_results(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    asr = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m['acc_distr'])
        asr.append(m.get('asr', 0.0)) # Use .get to avoid errors if a client didn't send it
        vcid.append(m['cid'])
    return {"acc_distr": acc, "asr": asr, "cid": vcid}

def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m.get('acc_val_distr', 0.0))
        vcid.append(m['cid'])
    return {"acc_val_distr": acc, "cid": vcid}