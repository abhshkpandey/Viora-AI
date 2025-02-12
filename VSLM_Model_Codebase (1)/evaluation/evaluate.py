import torch
from models.model import VSLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VSLMModel().to(device)
model.eval()

def evaluate():
    dummy_input = torch.randn(1, 768).to(device)
    output = model(dummy_input)
    print(f"Evaluation Output: {output}")

if __name__ == "__main__":
    evaluate()
