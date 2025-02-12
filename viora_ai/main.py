from model import get_model
from finetune import fine_tune_model
from train import load_finetuning_dataset, train_model

if __name__ == "__main__":
    model, tokenizer = get_model()
    dataset = load_finetuning_dataset(tokenizer)
    model = fine_tune_model(model)
    model = train_model(model, dataset)
    model.save_pretrained("./viora-ai")
    tokenizer.save_pretrained("./viora-ai")
    print("Training Complete! Model saved in './viora-ai'")
