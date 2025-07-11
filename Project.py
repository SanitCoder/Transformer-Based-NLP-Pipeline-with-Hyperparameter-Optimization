# 1. Data Loading
with open("wiki.train.tokens", "r", encoding="utf-8") as f:
    train_text = f.read()
with open("wiki.valid.tokens", "r", encoding="utf-8") as f:
    valid_text = f.read()
with open("wiki.test.tokens", "r", encoding="utf-8") as f:
    test_text = f.read()

# 2. Tokenizer Setup
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

def tokenize_text(text, tokenizer):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

train_encodings = tokenize_text(train_text, tokenizer)
valid_encodings = tokenize_text(valid_text, tokenizer)
test_encodings = tokenize_text(test_text, tokenizer)

# 3. Dataset Preparation
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],  # For language modeling, labels are the same as input_ids
        }

train_dataset = TextDataset(train_encodings)
valid_dataset = TextDataset(valid_encodings)
test_dataset = TextDataset(test_encodings)

# 4. Hyperparameter Optimization with Optuna
import optuna
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel

def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 4, 8])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01)
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 5)

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        eval_strategy='epoch',  # <-- Use eval_strategy as required by your version
        save_strategy='epoch',
        logging_strategy='epoch',
        disable_tqdm=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )

    # Load a new model for each trial
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_loss']

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

print("Best hyperparameters:", study.best_params)

# 5. Train Final Model with Best Hyperparameters
best_params = study.best_params

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    weight_decay=best_params['weight_decay'],
    num_train_epochs=best_params['num_train_epochs'],
    save_strategy='epoch',
    eval_strategy='epoch',  # <-- Use eval_strategy as required by your version
    logging_strategy='epoch',
    disable_tqdm=False,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()

# 6. Evaluation Utilities
import torch

def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

def top_k_accuracy(model, tokenizer, text, k=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    correct = 0
    total = 0

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for i in range(input_ids.shape[1] - 1):
            next_token_logits = logits[0, i]
            top_k_tokens = torch.topk(next_token_logits, k).indices
            true_token = input_ids[0, i + 1]
            if true_token in top_k_tokens:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0

# 7. Example Generation and Evaluation
# Evaluate on a manageable chunk of the validation set for realistic metrics
sample_valid_text = valid_text[:1024]
ppl = compute_perplexity(model, tokenizer, sample_valid_text)
print(f"Validation Perplexity: {ppl:.2f}")

acc = top_k_accuracy(model, tokenizer, sample_valid_text, k=5)
print(f"Validation Top-5 Accuracy: {acc:.2%}")

# Generation example (optional)
inputs = tokenizer(
    sample_valid_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024
)
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.2,
    max_length=30
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Sentence:", generated_text)
