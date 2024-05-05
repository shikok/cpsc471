from torch_geometric.datasets import MoleculeNet, BAShapes, BAMultiShapesDataset, BA2MotifDataset
import torch
from torch_geometric.data import Dataset

def binarize_labels(dataset):
    if dataset.num_classes == 2:
        for data in dataset:
            data.y = (data.y > 0.5).type(torch.float).view(-1)
        return dataset
    labels = torch.cat([data.y for data in dataset]).to(torch.int64)
    histogram = torch.bincount(labels)
    cumsum = torch.cumsum(histogram, dim=0)
    # Find the class that is closest to 50%
    split = torch.argmin(torch.abs(cumsum - len(labels) / 2))
    for data in dataset:
        data.y = (data.y > split).type(torch.float).view(-1)
    assert len(data.y) == len(labels)
    print(f"Splitting at {split} for dataset {dataset.__class__.__name__}")

# Example of how to use the factory function
def get_binary_dataset(dataset_name):
    # Fetch and prepare the dataset
    if dataset_name == 'BACE':
        dataset = MoleculeNet(root='data/BACE', name='BACE')
    elif dataset_name == 'BBBP':
        dataset = MoleculeNet(root='data/BBBP', name='BBBP')
    elif dataset_name == 'BAMultiShapesDataset':
        dataset = BAMultiShapesDataset(root='data/BAMultiShapesDataset')
    elif dataset_name == 'BA2Motif':
        dataset = BA2MotifDataset(root='data/BA2MotifDataset')
    else:
        raise ValueError('Invalid dataset name') # Maybe add BA2MotifDataset

    binarize_labels(dataset)
    return dataset
