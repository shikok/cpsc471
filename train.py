# Import necessary libraries
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from datasets import get_binary_dataset
from model import GCNStandardSupervised
from sklearn.metrics import roc_auc_score
import numpy as np

# Define training function
def train_model(model, loader, optimizer, device):
    model.train()
    model.to(device)
    total_loss = 0
    for data in loader:
        if (data.batch.max() + 1 != data.num_graphs):
            print("Skipping batch with different number of graphs")
            continue
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = model.get_loss_function(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Define evaluation function
def evaluate_model(model, loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    predictions = []
    targets = []
    for data in loader:
        if (data.batch.max() + 1 != data.num_graphs):
            print("Skipping batch with different number of graphs")
            continue
        data.to(device)
        out = model.predict(data.x, data.edge_index, data.batch)
        pred = (out.squeeze() > 0.5)
        target = (data.y.squeeze() > 0.5)
        correct += pred.eq(target).sum().item()
        total += data.num_graphs
        predictions.append(out.detach().cpu().numpy())
        targets.append(data.y.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    accuracy = correct / total
    auc = roc_auc_score(targets, predictions)
    return accuracy, auc

# Function to train on different datasets and pooling functions
def train_on_datasets(pooling_fn, datasets, gpu_ids=None, best_params=None):
    if gpu_ids is not None:
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for dataset_name in datasets:
        dataset = get_binary_dataset(dataset_name)
        num_features = getattr(dataset, 'num_features', dataset[0].num_features)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        if best_params is not None:
            num_conv_layers = best_params.get('num_conv_layers', 2)
            num_linear_layers = best_params.get('num_linear_layers', 2)
            hidden_size = best_params.get('hidden_size', 64)
            conv_channels = []
            for i in range(num_conv_layers):
                conv_channels.append(hidden_size)
            
            out_channels = []
            for i in range(num_linear_layers):
                out_channels.append(hidden_size)

            model = GCNStandardSupervised(
                in_channel=dataset.num_features,
                conv_channels=conv_channels,
                out_channels=out_channels,
                dropout=best_params.get('dropout', 0.1),
                pooling_fn=global_max_pool,
                loss_fn=torch.nn.BCEWithLogitsLoss,
                l1_penalty=0
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=best_params.get('lr', 0.01))
        else:
            model = GCNStandardSupervised(in_channel=num_features, 
                                                conv_channels=[32, 64], 
                                                out_channels=[64, 32], 
                                                pooling_fn=pooling_fn).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(50):
            train_loss = train_model(model, loader, optimizer, device)
            acc, auc = evaluate_model(model, loader, device)
            print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        # get the matching pooling function name from the dictionary
        torch.save(model, f"{dataset_name}_{pooling_fn}_full_model.pth")
        torch.save(model.state_dict(), f"{dataset_name}_{pooling_fn}.pt")
        results[f"{dataset_name}_{pooling_fn}"] = {'train_loss': train_loss, 'accuracy': acc}
    return results

if __name__ == '__main__':     

    # Define pooling functions
    # pooling_functions = {
    #     'mean': global_mean_pool,
    #     'max': global_max_pool,
    #     'sum': global_add_pool
    # }

    # # Define datasets
    # datasets = ['BAMultiShapesDataset', 'BACE', 'BBBP']
    pooling_functions = {
        'max': global_max_pool
    }
    datasets = ['BACE']
    results = train_on_datasets(pooling_functions, datasets)
    print(results)


# Placeholder for running the training function - uncomment in actual usage
# train_results = train_on_datasets(pooling_functions, datasets)

# Placeholder for results plotting - uncomment in actual usage
# plt.figure(figsize=(10, 8))
# for key, value in train_results.items():
#     plt.plot(value['train_loss'], label=f'{key} loss')
#     plt.plot(value['accuracy'], label=f'{key} accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Metric')
# plt.title('Training Performance')
# plt.legend()
# plt.show()

