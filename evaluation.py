import torch
import tqdm
from torch_geometric.loader import DataLoader
import numpy as np

def evaluate_model_with_importance(model, dataset, explanation_method, method_name, predictions, sigma_factor=0.2 , batch_size=1, is_NLE=False, zero_nodes=True):
    """
    Evaluates the model using various explanation methods.

    Args:
    - model: The trained GNN model.
    - dataset: The dataset to evaluate.
    - explanation_method: Function that calculates node importance or relevance.
    - method_name: A string to identify the explanation method in output.
    - sigma_factor: Hyperparameter to scale the perturbation magnitude.
    - batch_size: Batch size for data loading.

    Returns:
    - A dictionary of metric results.
    """
    metric_results = {"fidelity": [], "perturbation_impact": [], "accuracy": [], "stability": []}

    # Compute standard deviation of features across the entire dataset
    all_features = torch.cat([data.x for data in dataset])
    sigma = all_features.float().std(0, unbiased=False)

    per_dataset = []
    li_per_dataset = []
    for graph_idx, graph_data in enumerate(dataset):
        if graph_data.num_nodes == 0:
            continue
        importance_scores = explanation_method(model.to('cpu'), graph_data.to('cpu'), graph_idx)
        p_graph = perturb_graph(graph_data, importance_scores, predictions[graph_idx], sigma, sigma_factor, is_NLE=is_NLE, perturb_least_important=False, zero_nodes=zero_nodes)
        li_p_graph = perturb_graph(graph_data, importance_scores, predictions[graph_idx], sigma, sigma_factor, is_NLE=is_NLE, perturb_least_important=True, zero_nodes=zero_nodes)
        li_per_dataset.append(li_p_graph)
        per_dataset.append(p_graph)
        
    loader = DataLoader(per_dataset, batch_size=64, shuffle=False)
    loader_li = DataLoader(li_per_dataset, batch_size=64, shuffle=False)
    predicted_labels_perturbed = []
    predicted_prob_perturbed = []
    original_labels = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for graph_idx, graph_data in enumerate(tqdm.tqdm(loader)):
        graph_data.to(device)
        model.eval()
        prob = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        pred = (prob > 0.5).float()
        predicted_labels_perturbed.append(pred)
        predicted_prob_perturbed.append(prob)
        original_labels = original_labels + graph_data.y.detach().cpu().numpy().tolist()

    predicted_labels_perturbed_li = []
    
    for graph_idx, graph_data in enumerate(tqdm.tqdm(loader_li)):
        graph_data.to(device)
        model.eval()
        prob = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        pred = (prob > 0.5).float()
        predicted_labels_perturbed_li.append(pred)

    predicted_prob_perturbed = torch.cat(predicted_prob_perturbed).detach().cpu().numpy()
    predicted_labels_perturbed = torch.cat(predicted_labels_perturbed).detach().cpu().numpy()
    predicted_labels_perturbed_li = torch.cat(predicted_labels_perturbed_li).detach().cpu().numpy()
    original_labels = original_labels
    metric_results["fidelity"] = (predictions != predicted_labels_perturbed).mean()
    metric_results["perturbation_impact"] = np.abs(predicted_prob_perturbed - predictions).mean()
    metric_results["accuracy"] = (original_labels == predicted_labels_perturbed).mean()
    metric_results["stability"] = (predictions == predicted_labels_perturbed_li).mean()
    print(f"{method_name} - Evaluation Results:", metric_results)
    return metric_results



def perturb_graph(graph_data, importance_scores, prediction, sigma, sigma_factor, top_k=2, is_NLE=False, perturb_least_important=False, zero_nodes=True):
    perturbed_graph = graph_data.clone()
    if is_NLE and prediction < 0.5:
        importance_scores = {k: -v for k, v in importance_scores.items()}
    if perturb_least_important:
        top_k_nodes = sorted(importance_scores, key=importance_scores.get)[:top_k]
    else:    
        top_k_nodes = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_k]

    for node_idx in top_k_nodes:
        if zero_nodes:
            perturbed_graph.x[node_idx] = torch.zeros_like(perturbed_graph.x[node_idx])
        else:
            perturbed_graph.x[node_idx] = perturbed_graph.x[node_idx] + sigma_factor * sigma
    return perturbed_graph
