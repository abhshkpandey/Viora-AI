import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set PAD token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Define Mixture of Experts (MoE) Layer
class MoE(nn.Module):
    def __init__(self, num_experts=4, expert_dim=4096):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(expert_dim, expert_dim) for _ in range(num_experts)])
        self.gating = nn.Linear(expert_dim, num_experts)
    
    def forward(self, x):
        gate_weights = torch.softmax(self.gating(x), dim=-1)
        output = sum(w * expert(x) for w, expert in zip(gate_weights.T, self.experts))
        return output

# Replace Mistral's MLP layers with MoE
class MoETransformerLayer(nn.Module):
    def __init__(self, hidden_size=4096, num_experts=4):
        super().__init__()
        self.moe = MoE(num_experts, hidden_size)

    def forward(self, x):
        return self.moe(x)

for layer in model.model.layers:
    layer.mlp = MoETransformerLayer()

def get_model():
    return model, tokenizer
