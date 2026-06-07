import json
import torch
from torch_geometric.data import Data, Dataset

class CADGraphDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)
        self.part_names = list(self.raw_data.keys())

    def len(self):
        return len(self.part_names)

    def get(self, idx):
        part_name = self.part_names[idx]
        part_matrix = self.raw_data[part_name]
        
        # 1. Extract ALL 4 geometric features per face
        # Shape becomes [Num_Faces, 4]
        node_features = []
        for node in part_matrix["nodes"]:
            node_features.append(node["features"])
            
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # 2. Extract Link Topology
        edge_sources = []
        edge_targets = []
        for link in part_matrix["links"]:
            src = link["source"]
            tgt = link["target"]
            edge_sources.extend([src, tgt])
            edge_targets.extend([tgt, src])
            
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data.part_name = part_name
        
        return data

if __name__ == "__main__":
    dataset = CADGraphDataset("cad_graph_dataset.json")
    print(f"📚 Loaded PyTorch Geometric Dataset with {len(dataset)} enriched CAD graphs.")
    sample_part = dataset[0]
    print(f"📊 New Tensor X (Nodes, Features) Shape: {sample_part.x.shape} <-- Expecting 4 features!")