from datasets import load_dataset
from transformers import TrainingArguments, Trainer, default_data_collator

# Load dataset for fine-tuning
def load_finetuning_dataset(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    return dataset.map(tokenize_function, batched=True)

# Training Pipeline
def train_model(model, dataset):
    training_args = TrainingArguments(
        output_dir="./viora-ai",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    return model
