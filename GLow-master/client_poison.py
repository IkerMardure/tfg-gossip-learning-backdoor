from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from model import LeNet, train, test
from torch.utils.data import Dataset


def _resolve_torch_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"gpu", "h100", "cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        self.is_malicious = int(cid) in [2]  # Adibidea: Client 2 is malicious


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def poison_data(self, trainloader):
        # Label flipping de clase 0 a clase 1
        poisoned_dataset = PoisonedDataset(trainloader.dataset, from_class=0, to_class=1)
        return torch.utils.data.DataLoader(poisoned_dataset, batch_size=trainloader.batch_size, shuffle=True)
    
    def fit(self, parameters, config):
        # Copy params from server to local model
        self.set_parameters(parameters)
        metrics_val_distr = None

        # Perform local training
        if config['local_train_cid'] == self.cid or config['local_train_cid'] == -1:  # Case for GL or FL
            lr = config['lr']
            epochs = config['local_epochs']
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)

            # Poison data if the client is malicious
            print(f"Client {self.cid} is {'malicious' if self.is_malicious else 'benign'}.")
            if self.is_malicious:
                print(f"Client {self.cid} is malicious, poisoning data...")
                self.trainloader = self.poison_data(self.trainloader)

            # Local training
            distr_loss_train, metrics_val_distr = train(
                self.model, self.trainloader, self.validationloader, optim, epochs, self.num_classes, self.device
            )

            return self.get_parameters({}), len(self.trainloader), {'acc_val_distr': metrics_val_distr,'cid': self.cid, 'HEAD': 'YES', 'distr_val_loss': '##'}

        return self.get_parameters({}), len(self.trainloader), {
            'acc_val_distr': metrics_val_distr,
            'cid': self.cid,
            'energy used': '10W',
            'distr_val_loss': '##'
        }
    
    #Evaluate global model in validation set of a particular client
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.validationloader, self.num_classes, self.device)
        self.local_acc = accuracy
        return float(loss), len(self.validationloader), {'acc_distr': accuracy, 'cid': self.cid} #send anything, time it took to evaluation, memory usage...
    
    def get_local_acc(self):
        return self.local_acc   


def generate_client_fn(vcid, trainloaders, validationloaders, num_classes, device):
    def client_fn(cid: str):
        return FlowerClient(vcid[int(cid)], trainloader=trainloaders[int(cid)], validationloader=validationloaders[int(cid)], num_classes=num_classes, device=device).to_client()
    return client_fn

def cli_eval_distr_results(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m['acc_distr'])
        vcid.append(m['cid'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_distr": acc, "cid": vcid}

def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m['acc_val_distr'])
        vcid.append(m['cid'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_val_distr": acc, "cid": vcid}


class PoisonedDataset(Dataset):
    def __init__(self, dataset, from_class=0, to_class=1):
        self.dataset = dataset
        self.from_class = from_class
        self.to_class = to_class

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if y == self.from_class:
            y = self.to_class
        return x, y

    def __len__(self):
        return len(self.dataset)