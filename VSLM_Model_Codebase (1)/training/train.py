import torch
import torch.optim as optim
from models.model import VSLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VSLMModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train_step():
    model.train()
    optimizer.zero_grad()
    dummy_input = torch.randn(1, 768).to(device)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    print(f"Training step complete. Loss: {loss.item()}")

if __name__ == "__main__":
    train_step()
