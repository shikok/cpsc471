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

from torch_geometric.data import Data

def is_not_empty(data):
    # This function checks if the graph is not empty.
    return data.num_nodes > 0 and data.num_edges > 0

def get_binary_dataset(dataset_name):
    if dataset_name == 'BACE':
        dataset = MoleculeNet(root='data/BACE', name='BACE', pre_filter=is_not_empty)
    elif dataset_name == 'BBBP':
        dataset = MoleculeNet(root='data/BBBP', name='BBBP', pre_filter=is_not_empty)
    elif dataset_name == 'BAMultiShapesDataset':
        dataset = BAMultiShapesDataset(root='data/BAMultiShapesDataset')
    elif dataset_name == 'BA2Motif':
        dataset = BA2MotifDataset(root='data/BA2MotifDataset')
    else:
        raise ValueError('Invalid dataset name')

    # Binarize the labels of the dataset
    binarize_labels(dataset)

    return dataset

