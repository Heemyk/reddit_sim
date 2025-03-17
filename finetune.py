# Filename: finetune.py
from transformers import AutoModelForCausalLM, AutoTokenizer, SFTTrainer, TrainingArguments
from datasets import Dataset
import json
import torch

class RedditFinetuner:
    def __init__(self, model_name: str = "distilbert/distilgpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self, filename: str) -> Dataset:
        with open(filename, "r") as f:
            data = json.load(f)
        return Dataset.from_list(data)

    def format_example(self, example: dict) -> dict:
        text = f"### Instruction: {example['instruction']}\n### Response: {example['response']}"
        return {"text": text}

    def fine_tune(self, dataset: Dataset, output_dir: str = "finetuned_model"):
        dataset = dataset.map(self.format_example)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            learning_rate=2e-5,
        )
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
                "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
                "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
            }
        )
        trainer.train()
        trainer.save_model(output_dir)

if __name__ == "__main__":
    finetuner = RedditFinetuner()
    for user in ["user1", "user2"]:  # Replace with actual user IDs
        dataset = finetuner.load_data(f"data/finetune_data_{user}.json")
        finetuner.fine_tune(dataset, output_dir=f"finetuned_model_{user}")