import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
import torch

def explanation_method_gnnexplainer(model, graph_data, graph_idx):
    model.eval()
    batch = graph_data.batch if hasattr(graph_data, 'batch') else None

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type= None,
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        )
    )

    explanation = explainer(graph_data.x, graph_data.edge_index, batch=batch)
    node_importance_scores = torch.sigmoid(explanation.node_mask).detach().cpu().numpy()
    
    # Convert array to dictionary {node_index: node_score}
    importance_dict = {i: score for i, score in enumerate(node_importance_scores)}
    return importance_dict




def explanation_method_pgexplainer(model, graph_data, graph_idx):
    model.eval()
    batch = graph_data.batch if hasattr(graph_data, 'batch') else None

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=50, num_hops=3, lr=0.01),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type= None,
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        )
    )

    explanation = explainer(graph_data.x, graph_data.edge_index, batch=batch)
    node_importance_scores = explanation.node_probs.detach().cpu().numpy()

    # Convert array to dictionary {node_index: node_score}
    importance_dict = {i: score for i, score in enumerate(node_importance_scores)}
    return importance_dict



def explanation_method_gradcam(model, graph_data, graph_idx):
    model.eval()  # Ensure the model is in evaluation mode
    graph_data.x = graph_data.x.float()  # Convert x to float
    graph_data.x.requires_grad = True  # Set requires_grad to True

    # Forward pass
    out = model(graph_data.x, graph_data.edge_index, batch=graph_data.batch)
    out.backward(torch.ones_like(out))  # Backward pass to compute gradients

    # Compute the importance scores based on gradients
    gradients = graph_data.x.grad.abs()
    importance_scores = {i: gradients[i].sum().item() for i in range(graph_data.num_nodes)}

    return importance_scores



def explanation_method_random(model, graph_data, graph_idx):
    num_nodes = graph_data.x.size(0)
    random_scores = np.random.rand(num_nodes)

    # Convert array to dictionary {node_index: node_score}
    importance_dict = {i: score for i, score in enumerate(random_scores)}
    return importance_dict
