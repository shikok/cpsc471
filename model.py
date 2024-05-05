from typing import List, Union
import torch
from torch import Tensor
from torch.nn import Linear, Sigmoid, ModuleList, Dropout, ReLU, BatchNorm1d, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, pool 


class GCNStandardSupervised(torch.nn.Module):

    def __init__(self, 
            in_channel: int,
            conv_channels: List[int],
            out_channels: List[int],
            dropout: float = 0.1, batch_norm_args: dict = {'eps': 1e-5, 'momentum': 0.1}, 
            pooling_fn: pool = pool.global_mean_pool,
            loss_fn: torch.nn = BCEWithLogitsLoss, l1_penalty: Union[float, None] = None 
            ):
        super().__init__()
        
        # Convolutional layers:
        self.conv_layers = ModuleList()
        self.pass_edge_idx = list()
        for in_channel, out_channel in zip([in_channel, *conv_channels], conv_channels):
            self.conv_layers.extend([
                Dropout(p = dropout), 
                GCNConv(in_channel, out_channel),
                BatchNorm1d(out_channel, **batch_norm_args),
                ReLU()
                ])
            self.pass_edge_idx.extend([False, True, False, False])

        self.pooler = pooling_fn 
        
        self.lin_layers = ModuleList()
        for in_channel, out_channel in zip([conv_channels[-1], *out_channels], [*out_channels, 1]):
            self.lin_layers.extend([
                Dropout(p = dropout),
                Linear(in_channel, out_channel),
                BatchNorm1d(out_channel, **batch_norm_args),
                ReLU()
                ])
        self.lin_layers = self.lin_layers[:-1] # Remove final ReLU

        self.output_nonlin = Sigmoid()
        
        # Initialize loss function
        self.loss_function = loss_fn()
        self.l1_penalty = l1_penalty if l1_penalty is not None else 0

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.get_node_embeddings(x, edge_index)
        # Mean pool and predict
        x = self.pooler(x, batch)
        x = self.linear_classifier(x, batch)
        return x
    
    def linear_classifier(self, x: Tensor, batch: Tensor) -> Tensor:
        for op in self.lin_layers:
            # if batch is of size 1, skip the batch norm layer
            if isinstance(op, BatchNorm1d) and ( batch is None or batch.max() == 0):
                continue
            x = op(x)
        return x


    def predict(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.forward(x, edge_index, batch)
        return self.output_nonlin(x)
   
    def get_loss_function(self, predictions: Tensor, labels: Tensor) -> Tensor: 
        if predictions.shape != labels.shape:
           if predictions.dim() == 2 and predictions.size(1) == 1:
              predictions = predictions.squeeze(1) 
        labels = labels.to(predictions.dtype)
        loss = self.loss_function(predictions, labels)

        if self.l1_penalty > 0:  
            for p in self.parameters():
                loss += self.l1_penalty * torch.linalg.norm(torch.flatten(p), 1)
        
        return loss

    def get_node_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dtype == torch.long:
            x = x.float()
        # Compute graph convolutions
        for op, pass_ei in zip(self.conv_layers, self.pass_edge_idx):
            if pass_ei:
                x = op(x, edge_index)
            else:
                x = op(x)
        return x

