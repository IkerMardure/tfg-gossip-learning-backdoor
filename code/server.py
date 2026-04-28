from model import LeNet, test
import torch


def get_on_fit_config(config):
    def fit_config_fn(server_round: int):
        '''Decrease the learning rate from a specific communication round on'''
        #if server_round > 50:
        #    lr = config['lr'] / 10
        #else:
        #    lr = config['lr']
        fit_config = dict(config)
        fit_config['lr'] = config.get('lr', 0.001)
        fit_config['local_epochs'] = config.get('local_epochs', 1)
        fit_config['enable_tqdm'] = config.get('enable_tqdm', False)
        return fit_config
    
    return fit_config_fn

def get_evaluate_fn(num_classes: int, testloader):
    model = LeNet(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameter_keys = list(model.state_dict().keys())

    def evaluate_fn(server_round: int, parameters, config): #int nparrays, dict

        state_dict = model.state_dict()
        for key, value in zip(parameter_keys, parameters):
            state_dict[key].copy_(torch.as_tensor(value, device=state_dict[key].device, dtype=state_dict[key].dtype))
        model.load_state_dict(state_dict, strict=True)


        loss, accuracy = test(model, testloader, num_classes, device) #global model
        return loss, {'acc_cntrl': accuracy}

    return evaluate_fn
