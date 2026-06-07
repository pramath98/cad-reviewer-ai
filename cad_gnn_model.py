import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CADReviewerGCN(torch.nn.Module):
    def __init__(self):
        super(CADReviewerGCN, self).__init__()
        # Layer 1 now accepts all 4 geometric features (Area, Type, COM, Box Vol)
        self.conv1 = GCNConv(in_channels=4, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=32)
        
        self.classifier = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        graph_vector = global_mean_pool(x, batch)
        out = self.classifier(graph_vector)
        return torch.sigmoid(out)