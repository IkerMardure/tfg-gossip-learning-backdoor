import random
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from torch.utils.data import Dataset, DataLoader
# Make sure to import your model, train, and test functions
from model import LeNet, train, test
from utils.logging import log_client_training, log_data_poisoning 


BACKDOOR_POISON_RATE = 0.5
BACKDOOR_BOOST_FACTOR = 5.0


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
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.local_acc = None
        self.model = LeNet(num_classes)
        self.num_classes = num_classes
        self.device = _resolve_torch_device(device)
        self.is_malicious = int(cid) in [1]  # Client 1 is malicious

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def poison_data(self, trainloader):
        # Apply the backdoor trigger and change labels to class 0
        poisoned_dataset = BackdoorDataset(
            trainloader.dataset,
            target_class=0,
            poison_ratio=BACKDOOR_POISON_RATE,
        )
        return DataLoader(poisoned_dataset, batch_size=trainloader.batch_size, shuffle=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics_val_distr = None

        if config.get('local_train_cid', self.cid) == self.cid or config.get('local_train_cid') == -1:
            lr = config.get('lr', 0.001)
            epochs = config.get('local_epochs', 1)
            enable_tqdm = bool(config.get('enable_tqdm', False))
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)

            # Use logging module instead of print (level="verbose" for per-client detail)
            log_client_training(f"Client {self.cid} is {'malicious' if self.is_malicious else 'benign'}.", level="verbose")
            if self.is_malicious:
                log_data_poisoning(f"Client {self.cid} is malicious, poisoning data...", level="verbose")
                self.trainloader = self.poison_data(self.trainloader)

            # Local training
            progress_desc = f"cid {self.cid} - local train"
            distr_loss_train, metrics_val_distr = train(
                self.model,
                self.trainloader,
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

            return new_parameters, len(self.trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'distr_val_loss': '##'
            }

        return self.get_parameters({}), len(self.trainloader), {'cid': self.cid}
    
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