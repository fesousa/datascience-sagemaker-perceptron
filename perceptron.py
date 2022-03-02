import argparse
import json
import os
import numpy as np
import logging
import sys

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Perceptron(nn.Module):
    def __init__(self, num_features):
        super(Perceptron, self).__init__()
        self.num_features = num_features
        self.weight  = torch.nn.Parameter(torch.zeros(num_features,1).to(device))

    def forward(self, x):
        u = torch.matmul(self.weight.T, x)
        y = torch.where(u > 0., torch.ones(1), torch.ones(1)*0)
        return y
    
def get_data(data_dir):
    logger.info(f"Dados local {data_dir}")
    file = os.listdir(data_dir)[0] 
    data = np.genfromtxt(f'{data_dir}/{file}', delimiter=',')
    #categorias
    d = data[:, -1]

    # entrada do treinamento
    x = data[:, :-1]

    # adicionar atributo 1
    x0 = np.array([1]*len(x))
    x = np.insert(x, 0, x0, axis=1)
    x = torch.tensor(x.T, dtype=torch.float32, device=device)
    d = torch.reshape(torch.tensor(d, dtype=torch.float32, device=device),(1,len(d)))
    logger.info(f"Dados local {x}")
    return x,d

def train(args):
   
    # carregar dados
    x, d = get_data(args.data_dir_training)
    x_val, d_val = get_data(args.data_dir_validation)
    
    mse = 1.0
    n = 0
    q = len(x[0, :])

    model = Perceptron(len(x)).to(device)
    if device == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    
    model.train()
    while mse > args.error:
        # ativação interna do nerônio
        x_n = torch.reshape(x[:, n],(len(x),1))    
        y = model(x_n)

        #calcular novo peso        
        w2 = torch.mm(torch.reshape(args.a* (d[0,n]- y),(1,1)), torch.reshape(x_n,(1,len(x))))
        w = model.module.weight + w2.T

        model.module.weight = torch.nn.Parameter(w.to(device))       

        # fazer predição dos dados de x_val para calcular o erro
        p = model(x_val)
        
        # calcular erro médio quadrático      
        mse = ((d_val - p)**2).mean()
        logger.info(f"MSE {mse}")
        n = (n+1) % q

        
    logger.info(f"Pesos Final: {model.module.weight }")
    save_model(model, args.model_dir)


# carregar modelo
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Perceptron(0))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:        
        state_dict = torch.load(f)        
    model.module.weight  = torch.nn.Parameter(state_dict['module.weight'].to(device))
    return model.to(device)

# salvar modelo
def save_model(model, model_dir):
    logger.info(f"Salvando modelo. {model_dir}")
    logger.info(f'STATE: {model.cpu().state_dict()}')
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
   
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--a", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--error", type=float, default=0.1, metavar="M", help="Erro mínimo (default: 0.1)"
    )


    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir-validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())