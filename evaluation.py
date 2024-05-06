import torch
import tqdm
from torch_geometric.loader import DataLoader

def evaluate_model_with_importance(model, dataset, explanation_method, method_name, predictions, sigma_factor=0.2 , batch_size=1, is_NLE=False):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metric_results = {"fidelity": [], "perturbation_impact": [], "accuracy": [], "stability": []}

    # Compute standard deviation of features across the entire dataset
    all_features = torch.cat([data.x for data in dataset])
    sigma = all_features.float().std(0, unbiased=False)

    for graph_idx, graph_data in enumerate(tqdm.tqdm(loader)):
        if graph_data.num_nodes == 0:
            continue
        original_pred = predictions[graph_idx]
        importance_scores = explanation_method(model, graph_data, graph_idx)
        perturbed_graph = perturb_graph(graph_data, importance_scores, predictions[graph_idx], sigma, sigma_factor, is_NLE=is_NLE)
        perturbed_pred = model.predict(perturbed_graph.x, perturbed_graph.edge_index, perturbed_graph.batch).detach()
        perturbed_pred_label = (perturbed_pred > 0.5).float()

        # Evaluate metrics
        fidelity = torch.mean((original_pred == perturbed_pred_label).float()).item()
        perturbation_impact = torch.mean(torch.abs(original_pred - perturbed_pred)).item()
        accuracy = torch.mean((original_pred == graph_data.y).float()).item()
        less_important_perturbed_graph = perturb_graph(graph_data, importance_scores, predictions[graph_idx], sigma, sigma_factor, is_NLE=is_NLE, perturb_least_important=True)
        less_important_perturbed_pred = model.predict(less_important_perturbed_graph.x, less_important_perturbed_graph.edge_index, less_important_perturbed_graph.batch).detach()
        stability = torch.mean(torch.abs(original_pred - less_important_perturbed_pred)).item()

        metric_results["fidelity"].append(fidelity)
        metric_results["perturbation_impact"].append(perturbation_impact)
        metric_results["accuracy"].append(accuracy)
        metric_results["stability"].append(stability)

    for key in metric_results:
        metric_results[key] = sum(metric_results[key]) / len(metric_results[key])

    print(f"{method_name} - Evaluation Results:", metric_results)
    return metric_results

def perturb_graph(graph_data, importance_scores, prediction, sigma, sigma_factor, top_k=5, is_NLE=False, perturb_least_important=False):
    perturbed_graph = graph_data.clone()
    if is_NLE and prediction < 0.5:
        importance_scores = {k: -v for k, v in importance_scores.items()}
    if perturb_least_important:
        top_k_nodes = sorted(importance_scores, key=importance_scores.get)[:top_k]
    else:    
        top_k_nodes = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_k]
    
    perturbed_graph.x = perturbed_graph.x.float()
    for node in top_k_nodes:
        noise = torch.randn(perturbed_graph.x[node].size()) * sigma * sigma_factor
        perturbed_graph.x[node] += noise

    return perturbed_graph
