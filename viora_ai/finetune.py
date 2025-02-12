from peft import LoraConfig, get_peft_model

# Fine-Tune with LoRA
def fine_tune_model(model):
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    return model
