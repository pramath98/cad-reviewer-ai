import torch
import torch.nn as nn
from dataset_loader import CADGraphDataset
from cad_gnn_model import CADReviewerGCN

# 1. Initialize Dataset & Hardware Acceleration
dataset = CADGraphDataset("cad_graph_dataset.json")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CADReviewerGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print(f"🚀 Starting Training Loop on device: {device}...")

# 2. Training Loop over multiple Epochs
model.train()
for epoch in range(1, 21): # Train for 20 iterations
    total_loss = 0
    
    for i in range(len(dataset)):
        data = dataset[i].to(device)
        
        # Generate a synthetic label on the fly based on geometric complexity
        # If links-to-nodes ratio is tight, it's a simple part (0), otherwise complex (1)
        ratio = data.edge_index.size(1) / data.x.size(0)
        label = torch.tensor([[1.0]], device=device) if ratio > 4.0 else torch.tensor([[0.0]], device=device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass prediction
        prediction = model(data)
        
        # Compute loss error and propagate back down the graph
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch:02d}/20 | Avg Graph Error Loss: {round(total_loss / len(dataset), 4)}")

print("\n🎉 Training run complete! Your M5 just optimized its first GNN weights.")