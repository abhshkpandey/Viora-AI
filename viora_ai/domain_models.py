from transformers import AutoModelForCausalLM

# Load Ultra-Small Domain-Specific Models
ultra_small_models = {
    "code": "codellama/CodeLlama-7B-Python",
    "finance": "FinGPT/FinGPT-6B",
    "medical": "MedAlpaca/MedAlpaca-7B",
    "general": "microsoft/phi-2"
}

def load_ultra_small_models():
    loaded_models = {}
    for domain, model_path in ultra_small_models.items():
        loaded_models[domain] = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return loaded_models
