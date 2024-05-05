import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_geometric.loader import DataLoader


class GraphNodeImportance:
    """
    Class to calculate node importance scores for a given graph dataset.
    @param model: The model used to calculate the graph embeddings.
    @param train_dataset: The dataset used to train the model.
    @param p: The exponent used in the node importance calculation.
    @param reload_save_path: The path to save or load the graph embeddings.
    """
    def __init__(self, model: torch.nn.Module, train_dataset: InMemoryDataset, p: int = 3, reload_save_path: str = None, device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.train_dataset = train_dataset
        self.device = device
        self.p = p
        self.graph_embeddings = None
        self.pred_labels = None
        self.graph_node_embeddings_dict = None
        self.importence_scores = None

        if reload_save_path is not None and Path(reload_save_path).exists():
            self.load_graph_embeddings(reload_save_path)
        else:
            self._calculate_graph_embeddings_with_labels()
            if reload_save_path is not None:
                self.save_graph_embeddings(reload_save_path)

    def _calculate_graph_embeddings_with_labels(self):
        graph_embeddings = None
        pred_labels = np.array([])
        loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False)
        graph_node_embeddings_dict = {}
        graph_running_id = 0
        for graph_data in loader:
            graph_data.to(self.device)
            self.model.eval()
            node_embd = self.model.get_node_embeddings(graph_data.x, graph_data.edge_index)

            # Initialize a temporary dictionary for node embeddings in each graph of the batch
            for graph_id in range(graph_data.num_graphs):
                graph_node_embeddings_dict[graph_running_id + graph_id] = node_embd[graph_data.batch == graph_id].detach().cpu().numpy()

            graph_embedding = self.model.pooler(node_embd, graph_data.batch)
            pred_prob = self.model.predict(graph_data.x, graph_data.edge_index, graph_data.batch)
            pred_label = (pred_prob.squeeze() > 0.5).float()
            graph_embedding_batch = graph_embedding.squeeze().detach().cpu().numpy()
            pred_label_batch = pred_label.squeeze().detach().cpu().numpy()

            # graph_embeddings is a 2D numpy array with shape (num_graphs, embedding_dim)
            graph_embeddings = graph_embedding_batch if graph_embeddings is None else np.append(graph_embeddings, graph_embedding_batch, axis=0)
            pred_labels = np.append(pred_labels, pred_label_batch)
            graph_running_id += graph_data.num_graphs

        self.graph_embeddings = graph_embeddings
        self.pred_labels = pred_labels
        self.graph_node_embeddings_dict = graph_node_embeddings_dict
    
    def _get_node_embeddings(self, graph_id):
        return self.graph_node_embeddings_dict[graph_id]

    def calculate_node_importance(self, graph, graph_id, return_graph=False):
        node_embeddings = self._get_node_embeddings(graph_id)
        if self.importence_scores is not None:
            importance_scores = self.importence_scores[graph_id]
        else:
            importance_scores = {}
            for i, node_embedding in enumerate(node_embeddings):
                numerator = 0
                denominator = 0
                for g, label in zip(self.graph_embeddings, self.pred_labels):
                    w_label = 1 if label == 1 else -1
                    distance = np.linalg.norm(g - node_embedding)
                    norm_dist = distance ** -self.p
                    weight = norm_dist * w_label
                    numerator += weight
                    denominator += norm_dist
                importance = numerator / denominator if denominator != 0 else 0
                importance_scores[i] = importance
        if not return_graph:
            return importance_scores
        graph_importance = graph.clone()
        graph_importance.x = torch.tensor(list(importance_scores.values())).unsqueeze(1)
        return graph_importance

    def save_graph_embeddings(self, path):
        np.save(path+ f"{self.p}_graph_embeddings.npy", self.graph_embeddings)
        np.save(path+ f"{self.p}_pred_labels.npy", self.pred_labels)
        np.save(path+ f"{self.p}_graph_node_embeddings_dict.npy", self.graph_node_embeddings_dict)

    def load_graph_embeddings(self, path):
        self.graph_embeddings = np.load(path+ f"{self.p}_graph_embeddings.npy")
        self.pred_labels = np.load(path+ f"{self.p}_pred_labels.npy")
        self.graph_node_embeddings_dict = np.load(path+ f"{self.p}_graph_node_embeddings_dict.npy")

        
    def visualize_graph_with_importance(self, graph, graph_id):
        G = nx.Graph()
        importance_scores = self.calculate_node_importance(graph, graph_id, return_graph=False)
        importance_scores = [importance_scores[node] for node in sorted(importance_scores.keys())]
        nodes = range(len(importance_scores))

        G.add_nodes_from(nodes)
        G.add_edges_from(graph.edge_index.t().numpy())
        norm = Normalize(vmin=-1, vmax=1)
        cmap = plt.cm.RdBu
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        node_colors = mappable.to_rgba(importance_scores)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=node_colors, node_size=200,
                with_labels=True, font_weight='bold', edge_color='black')
        plt.title('Graph with Node Importance')
        plt.colorbar(mappable, ax=plt.gca(), orientation='vertical', label='Node Importance')
        plt.show()

    def visualize_graph_embedding_with_2d_tsne(self):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        graph_embeddings = np.array([g.squeeze() for g in self.graph_embeddings])
        embeddings_2d = tsne.fit_transform(graph_embeddings)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.pred_labels, cmap='coolwarm')
        plt.title('Graph Embeddings with 2D t-SNE')
        plt.colorbar()
        plt.show()

    def calculate_all_node_importance(self):
        all_importance_scores = {}
        for graph_id in self.graph_node_embeddings_dict.keys():
            all_importance_scores[graph_id] = self.calculate_node_importance(None, graph_id)
        self.importence_scores = all_importance_scores
    
    def histogram_node_importance_distribution(self):
        if self.importence_scores is None:
            self.calculate_all_node_importance()
        # roll out the dict of dicts into a single list
        all_dicts = [d for d in self.importence_scores.values()]
        all_values = [v for d in all_dicts for v in d.values()]
        plt.hist(all_values, bins=20)
        plt.title('Node Importance Distribution')
        plt.show()

    def visualize_all_node_importance_distribution(self, add_graph_embedding=False):
        # use UMAP to reduce the dimensionality of the node embeddings
        import umap
        umap = umap.UMAP(n_components=2, random_state=42)
        all_node_embeddings = np.concatenate(list(self.graph_node_embeddings_dict.values()), axis=0)
        embeddings_2d = umap.fit_transform(all_node_embeddings)
        if self.importence_scores is None:
            self.calculate_all_node_importance()
        all_dicts = [d for d in self.importence_scores.values()]
        all_values = [v for d in all_dicts for v in d.values()]
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_values, cmap='coolwarm', s=1)
        plt.title('All Node Importance Distribution with 2D UMAP')
        plt.colorbar()
        if add_graph_embedding:
            plt.scatter(self.graph_embeddings[:, 0], self.graph_embeddings[:, 1], c=self.pred_labels, cmap='coolwarm', marker='x', s=10)
        plt.show()

    def explanation_method(self, graph_data, graph_idx):
        # This method integrates with the evaluation framework
        return self.calculate_node_importance(graph_data, graph_idx)
