import optuna
import torch
from torch import optim
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from model import GCNStandardSupervised
from datasets import get_binary_dataset
from train import train_model, evaluate_model
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from datasets import get_binary_dataset



def objective(trial, dataset_name):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = 64

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = get_binary_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Suggest the number of convolutional layers and their corresponding channels
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 4)
    num_linear_layers = trial.suggest_int('num_linear_layers', 1, 2)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    conv_channels = []
    for i in range(num_conv_layers):
        conv_channels.append(hidden_size)
    
    out_channels = []
    for i in range(num_linear_layers):
        out_channels.append(hidden_size)

    # Model initialization
    num_features = dataset.num_features
    model = GCNStandardSupervised(
        in_channel=num_features,
        conv_channels=conv_channels,
        out_channels=out_channels, 
        dropout=dropout,
        pooling_fn=global_max_pool,
        loss_fn=torch.nn.BCEWithLogitsLoss,
        l1_penalty=0
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training and Evaluation
    max_auc = 0
    for epoch in range(50):
        train_model(model, loader, optimizer, device)
        accuracy, auc = evaluate_model(model, loader, device)
        max_auc = max(max_auc, auc)
        trial.report(auc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return max_auc

