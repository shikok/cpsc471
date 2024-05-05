import torch
import tqdm
from torch_geometric.data import DataLoader

def evaluate_model_with_importance(model, dataset, explanation_method, method_name, predictions, batch_size=1, is_NLE=False):
    """
    Evaluates the model using various explanation methods.

    Args:
    - model: The trained GNN model.
    - dataset: The dataset to evaluate.
    - explanation_method: Function that calculates node importance or relevance.
    - method_name: A string to identify the explanation method in output.
    - batch_size: Batch size for data loading.

    Returns:
    - A dictionary of metric results.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Stores results for evaluation metrics
    metric_results = {
        "fidelity": [],
        "perturbation_impact": [],
        "accuracy": [],
        "stability": []
    }

    for graph_idx, graph_data in enumerate(tqdm.tqdm(loader)):
        # Original model prediction
        original_pred = predictions[graph_idx]

        # Calculate node importance using the provided method
        importance_scores = explanation_method(model, graph_data, graph_idx)

        # Perturb the graph based on importance scores
        perturbed_graph = perturb_graph(graph_data, importance_scores, predictions[graph_idx], is_NLE=is_NLE)

        # Model prediction on perturbed graph
        perturbed_pred = model.predict(perturbed_graph.x, perturbed_graph.edge_index, perturbed_graph.batch).detach()
        perturbed_pred_label = (perturbed_pred > 0.5).float()

        # Evaluation metrics
        fidelity = torch.mean((original_pred == perturbed_pred_label).float()).item()
        perturbation_impact = torch.mean(torch.abs(original_pred - perturbed_pred)).item()
        accuracy = torch.mean((original_pred == graph_data.y).float()).item()

        # Calculating stability by perturbing less important nodes
        less_important_perturbed_graph = perturb_graph_least_important(graph_data, importance_scores, predictions[graph_idx], is_NLE=is_NLE)
        less_important_perturbed_pred = model.predict(less_important_perturbed_graph.x, less_important_perturbed_graph.edge_index, less_important_perturbed_graph.batch).detach()
        stability = torch.mean(torch.abs(original_pred - less_important_perturbed_pred)).item()

        # Collect results
        metric_results["fidelity"].append(fidelity)
        metric_results["perturbation_impact"].append(perturbation_impact)
        metric_results["accuracy"].append(accuracy)
        metric_results["stability"].append(stability)

    # Average results
    for key in metric_results:
        metric_results[key] = sum(metric_results[key]) / len(metric_results[key])

    print(f"{method_name} - Evaluation Results:", metric_results)
    return metric_results



def perturb_graph(graph_data, importance_scores, prediction, top_k=5, is_NLE=False):
    # Clone the graph to avoid modifying the original
    perturbed_graph = graph_data.clone()
    graph_labels = prediction
    # if graph_labels == 1 most important nodes are with positive importance scores and vice versa
    # Find the top-k important nodes
    if graph_labels == 0 and is_NLE:
        top_k_nodes = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_k]
    else:    
        top_k_nodes = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_k]

    # Perturb the features of the top-k nodes (e.g., set to zero or random noise)
    for node in top_k_nodes:
        perturbed_graph.x[node] = torch.randn(perturbed_graph.x[node].size())

    return perturbed_graph

def perturb_graph_least_important(graph_data, importance_scores, prediction, bottom_k=5, is_NLE=False):
    # Clone the graph to avoid modifying the original
    perturbed_graph = graph_data.clone()
    graph_labels = prediction
    # Find the bottom-k least important nodes
    if graph_labels == 0 and is_NLE:
        bottom_k_nodes = sorted(importance_scores, key=importance_scores.get, reverse=True)[:bottom_k]
    else:
        bottom_k_nodes = sorted(importance_scores, key=importance_scores.get)[:bottom_k]

    # Perturb the features of the bottom-k nodes (e.g., set to zero or random noise)
    for node in bottom_k_nodes:
        perturbed_graph.x[node] = torch.randn(perturbed_graph.x[node].size())

    return perturbed_graph


# def evaluate_model_with_importance(model, dataset, importance_calculator, batch_size=1):
#     # Data loader for processing batches of data
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Stores results for evaluation metrics
#     metric_results = {
#         "fidelity": [],
#         "perturbation_impact": [],
#         "accuracy": [],
#         "stability": []
#     }

#     for graph_data in tqdm.tqdm(loader):
#         # Original model prediction
#         original_pred = model.predict(graph_data.x, graph_data.edge_index, graph_data.batch).detach()
#         original_pred_label = (original_pred > 0.5).float()

#         # Calculate node importance
#         importance_scores = importance_calculator.calculate_node_importance(graph_data)
#         importance_scores = explanation_method(model, graph_data)


#         # Perturb the graph based on importance scores
#         perturbed_graph = perturb_graph(graph_data, importance_scores)

#         # Model prediction on perturbed graph
#         perturbed_pred = model.predict(perturbed_graph.x, perturbed_graph.edge_index, perturbed_graph.batch).detach()
#         perturbed_pred_label = (perturbed_pred > 0.5).float()

#         # Evaluation metrics
#         fidelity = torch.mean((original_pred_label == perturbed_pred_label).float()).item()
#         perturbation_impact = torch.mean(torch.abs(original_pred - perturbed_pred)).item()
#         accuracy = torch.mean((original_pred_label == graph_data.y).float()).item()

#         # Calculating stability by perturbing less important nodes
#         less_important_perturbed_graph = perturb_graph_least_important(graph_data, importance_scores)
#         less_important_perturbed_pred = model.predict(less_important_perturbed_graph.x, less_important_perturbed_graph.edge_index, less_important_perturbed_graph.batch).detach()
#         stability = torch.mean(torch.abs(original_pred - less_important_perturbed_pred)).item()

#         # Collect results
#         metric_results["fidelity"].append(fidelity)
#         metric_results["perturbation_impact"].append(perturbation_impact)
#         metric_results["accuracy"].append(accuracy)
#         metric_results["stability"].append(stability)

#     # Average results
#     for key in metric_results:
#         metric_results[key] = sum(metric_results[key]) / len(metric_results[key])

#     return metric_results