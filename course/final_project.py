import os
import random
import re
import numpy as np
import torch
import PyPDF2
import evaluate
from huggingface_hub import notebook_login

# Datasets & Transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
# For LoRA/PEFT
from peft import LoraConfig, TaskType, get_peft_model

#############################################################################
#                            PDF -> Local Dataset
#############################################################################

def extract_text_from_pdf(pdf_path):
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
    return "\n".join(text_pages)

def naive_find_abstract_and_body(raw_text):
    """
    Very naive approach:
      (abstract, body) based on 'Abstract' heading
    """
    lines = raw_text.splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    inside_abstract = False
    abstract_lines = []
    body_lines = []

    abstract_pat = re.compile(r"^\s*Abstract\b", re.IGNORECASE)
    next_section_pat = re.compile(r"^\s*(introduction|1\.|2\.|keywords)\b", re.IGNORECASE)

    for line in lines:
        if not inside_abstract:
            if abstract_pat.match(line):
                inside_abstract = True
                if line.lower().strip() != "abstract":
                    abstract_lines.append(line)
            else:
                body_lines.append(line)
        else:
            if next_section_pat.match(line):
                inside_abstract = False
                body_lines.append(line)
            else:
                abstract_lines.append(line)

    abstract_txt = " ".join(abstract_lines).strip()
    body_txt = " ".join(body_lines).strip()
    return (abstract_txt if abstract_txt else "", body_txt if body_txt else "")

def build_examples_from_pdfs(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    data = []
    for pdf_file in pdf_files:
        path = os.path.join(pdf_folder, pdf_file)
        raw_text = extract_text_from_pdf(path)
        abstract, body = naive_find_abstract_and_body(raw_text)
        data.append({
            "filename": pdf_file,
            "body": body,
            "summary": abstract
        })
    return data

def split_train_val_test(examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = examples[:train_end]
    val_data   = examples[train_end:val_end]
    test_data  = examples[val_end:]

    dset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    return dset

def maybe_build_and_push_local_dataset(pdf_folder, dataset_repo_id):
    """
    If "local_pdf_dataset" folder doesn't exist, parse PDFs -> create local dataset -> push.
    Otherwise, skip PDF processing.
    """
    if os.path.exists("local_pdf_dataset"):
        print("local_pdf_dataset folder exists. Skipping PDF parse & dataset creation.")
        return
    examples = build_examples_from_pdfs(pdf_folder)
    if len(examples) == 0:
        print("No PDFs found or no data extracted. Exiting.")
        return
    dataset = split_train_val_test(examples)
    print("Created dataset splits:", {k: len(dataset[k]) for k in dataset})
    dataset.save_to_disk("local_pdf_dataset")
    dataset.push_to_hub(dataset_repo_id)
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_repo_id}")

#############################################################################
#                           Summarization Functions
#############################################################################

def preprocess_function(examples, tokenizer, body_key, summary_key, max_input_len=512, max_target_len=128):
    model_inputs = tokenizer(examples[body_key], truncation=True, max_length=max_input_len)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[summary_key], truncation=True, max_length=max_target_len)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_rouge_scores(model, dataset, tokenizer, device, body_key="body", summary_key="summary", max_length=128, num_beams=4):
    """
    Evaluate a model by generating from 'body_key' -> compare with 'summary_key'
    """
    rouge = evaluate.load("rouge")
    preds, refs = [], []

    for ex in dataset:
        body_text = ex[body_key]
        ref_text  = ex[summary_key]
        if not body_text.strip():
            preds.append("")
            refs.append(ref_text)
            continue

        input_ids = tokenizer.encode(body_text, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(pred_text)
        refs.append(ref_text)

    result = rouge.compute(predictions=preds, references=refs)
    if isinstance(result["rouge1"], float):
        return {k: v*100 for k, v in result.items()}
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}


def train_lora(base_model, dataset, tokenizer, model_repo_id, body_key="body", summary_key="summary", num_epochs=2):
    """
    1) Wrap base_model in LoRA
    2) Fine-tune on 'dataset'
    3) Return the final LoRA model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    base_model.to(device)

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q","v"]  # or q_proj/v_proj if T5 uses that
    )
    lora_model = get_peft_model(base_model, peft_config).to(device)

    # Preprocess
    def token_map_fn(examples):
        return preprocess_function(examples, tokenizer, body_key, summary_key)

    tokenized_ds = dataset.map(token_map_fn, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=lora_model,
        label_pad_token_id=-100
    )

    # Metric
    rouge = evaluate.load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # shape fix
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = np.squeeze(preds, axis=1)
        if labels.ndim == 3 and labels.shape[1] == 1:
            labels = np.squeeze(labels, axis=1)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        if isinstance(result["rouge1"], float):
            return {k: v*100 for k,v in result.items()}
        return {k: v.mid.fmeasure * 100 for k,v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="model_lora_temp",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id=model_repo_id,
        hub_strategy="end",
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print(f"\n=== Start LoRA Fine-tuning on {model_repo_id} ===")
    trainer.train()
    print("=== LoRA Fine-tuning done ===")

    # Evaluate on test set
    final_eval = trainer.evaluate(tokenized_ds["test"])
    print("Trainer Evaluate (test set):", final_eval)
    return lora_model


#############################################################################
#                                  MAIN
#############################################################################

def main():
    pdf_folder = "sources"
    dataset_repo_id = "benitoals/my-pdf-dataset"
    huggingface_science = "armanc/scientific_papers"
    model_name = "google/mt5-small"

    # We'll create new repos for each stage, or you can reuse
    local_model_repo_id = "benitoals/my-lora-local"
    hf_model_repo_id    = "benitoals/my-lora-hf"
    combined_repo_id    = "benitoals/my-lora-combined"

    # Step A: If local dataset not built, build from PDF & push
    maybe_build_and_push_local_dataset(pdf_folder, dataset_repo_id)

    # 1) Load local dataset
    local_data = load_dataset(dataset_repo_id)
    print("Local dataset loaded:", local_data)

    # 2) Baseline: no training => evaluate on local_data["test"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    baseline_model.to(device)

    baseline_rouge = get_rouge_scores(baseline_model, local_data["test"], tokenizer, device)
    print("\n=== Baseline Zero-shot on local test ===")
    print(baseline_rouge)

    # 3) Fine-tune on local dataset
    #    Then evaluate on local test
    local_trained_model = train_lora(baseline_model, local_data, tokenizer, local_model_repo_id, "body", "summary", num_epochs=2)
    local_trained_model.eval()
    local_after_rouge = get_rouge_scores(local_trained_model, local_data["test"], tokenizer, device)
    print("\n=== After training on local dataset (LoRA) => local test ===")
    print(local_after_rouge)

    # 4) Baseline model again => fine-tune on huggingface_science => evaluate on local test
    print("\n=== Fine-tune on huggingface_science ===")
    baseline_model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # We'll load the "armanc/scientific_papers" dataset: "train", "validation", "test"
    # For summarization: "article" -> "abstract"
    scipapers = load_dataset(huggingface_science, split={"train":"train","validation":"validation","test":"test"})
    # We'll do a minimal example, or you can choose a subset for speed
    # e.g. scipapers["train"] = scipapers["train"].select(range(500)) # optional subset

    # Fine-tune baseline_model2 on scipapers
    # We'll define a quick function to convert it into a DatasetDict with train/val/test
    scipapers_dset = DatasetDict({
        "train": scipapers["train"],
        "validation": scipapers["validation"],
        "test": scipapers["test"],
    })
    hf_trained_model = train_lora(baseline_model2, scipapers_dset, tokenizer, hf_model_repo_id, "article", "abstract", num_epochs=1)
    hf_trained_model.eval()

    # Evaluate that model on local test
    hf_on_local_rouge = get_rouge_scores(hf_trained_model, local_data["test"], tokenizer, device)
    print("\n=== After training on HuggingFace scientific dataset => local test ===")
    print(hf_on_local_rouge)

    # 5) Now "boost" from that model => train on local dataset => evaluate
    final_model = train_lora(hf_trained_model, local_data, tokenizer, combined_repo_id, "body", "summary", num_epochs=2)
    final_model.eval()
    final_rouge = get_rouge_scores(final_model, local_data["test"], tokenizer, device)
    print("\n=== After HF dataset + local => local test ===")
    print(final_rouge)

    # Finally print all four:
    # baseline_rouge
    # local_after_rouge
    # hf_on_local_rouge
    # final_rouge
    print("\n===== All four results =====")
    print("1) Baseline Zero-shot on local test       =>", baseline_rouge)
    print("2) LoRA on local dataset => local test    =>", local_after_rouge)
    print("3) LoRA on huggingface_science => local test =>", hf_on_local_rouge)
    print("4) HF + local => local test =>", final_rouge)

if __name__ == "__main__":
    main()