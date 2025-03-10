# %% [markdown]
# # Summarization Fine-Tuning Sandbox
# 
# Code adapted from the Fine-tuning Sandbox example (Shawhin Talebi) to perform summarization.
# This notebook fine-tunes an mT5 model to summarize review bodies into review titles.
# %%
from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

# Load the English and Spanish splits of the Amazon Reviews Multi dataset
english_dataset = load_dataset("mteb/amazon_reviews_multi", "en")
spanish_dataset = load_dataset("mteb/amazon_reviews_multi", "es")

# %%
# Split the raw text into review_title and review_body
def split_review(example):
    # Split on a double newline; adjust delimiter if necessary
    parts = example["text"].split("\n\n", 1)
    if len(parts) == 2:
        review_title, review_body = parts
    else:
        review_title = parts[0]
        review_body = ""
    # Return a dictionary with the new fields
    return {"review_title": review_title, "review_body": review_body}


english_dataset = english_dataset.map(split_review)
spanish_dataset = spanish_dataset.map(split_review)

# %%
# Filter to keep only examples with label == 4 (as in your original code)
def filter_types(example):
    return (
        example["label"] == 4
    )

english_type_4 = english_dataset.filter(filter_types)
spanish_type_4 = spanish_dataset.filter(filter_types)

# %%
# Concatenate English and Spanish datasets into a single DatasetDict
books_dataset = DatasetDict()

for split in english_type_4.keys():
    books_dataset[split] = concatenate_datasets(
        [english_type_4[split], spanish_type_4[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=13213)

# Optionally filter out examples with very short titles
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)

print(books_dataset["train"][0]["review_title"])  # Expect a single string
print(books_dataset["train"][0]["review_body"])   # Expect a single string

# %% [markdown]
# ### Model

# %%
# Choose the mT5-small checkpoint and load the fast tokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# Load the summarization model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))



# %% [markdown]
# ### Preprocess Data

# %%
# Set maximum lengths for inputs (review_body) and targets (review_title)
max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
        padding="longest"
    )
    labels = tokenizer(
        examples["review_title"],
        max_length=max_target_length,
        truncation=True,
        padding="longest"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
columns_to_remove = books_dataset["train"].column_names
tokenized_datasets = books_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=columns_to_remove
)
# Create a data collator that will dynamically pad (if needed) during training.
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100  # Ensures labels are padded correctly
)

# %% [markdown]
# ### Evaluation

# %%
import evaluate
rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# %% [markdown]
# ### Train Model

# %%
# for name, module in model.named_modules():
#     # Print lines that match any "q"/"k"/"v"/"o"
#     if any(x in name.lower() for x in ["q", "k", "v", "o"]):
#         print(name)

# Set up LoRA fine-tuning configuration for sequence-to-sequence tasks
peft_config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=["q", "v"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Hyperparameters
lr = 1e-3
batch_size = 8
num_epochs = 3

training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-summarization",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="tensorboard"
)
# 1. Pick the right device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU.")

model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# %% [markdown]
# ### Generate Prediction

# %%

print("Trained model summaries:")
print("------------------------")
for i in range(5):
    text = books_dataset["test"][i]["review_body"]
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
    outputs = model.generate(inputs, max_length=max_target_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Review {i} Summary: {summary}")

# %% [markdown]
# ### Optional: Push Model to Hub

# %%
from huggingface_hub import notebook_login
notebook_login()  # Ensure token gives write access

hf_name = "benitoals"  
model_id = hf_name + "/" + model_checkpoint + "-lora-summarization"

model.push_to_hub(model_id)
trainer.push_to_hub(model_id)