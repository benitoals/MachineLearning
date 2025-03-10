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
#                           PDF -> Dataset
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
    train_end = int(n*train_ratio)
    val_end = int(n*(train_ratio + val_ratio))
    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

def maybe_build_and_push_local_dataset(pdf_folder, dataset_repo_id):
    """If local_pdf_dataset/ exists, skip re-parsing. Otherwise parse & push."""
    if os.path.exists("local_pdf_dataset"):
        print("local_pdf_dataset folder exists. Skipping PDF parse & dataset creation.")
        return
    examples = build_examples_from_pdfs(pdf_folder)
    if not examples:
        print("No PDFs found or no data extracted. Exiting.")
        return
    dataset = split_train_val_test(examples)
    print("Created dataset splits:", {k: len(dataset[k]) for k in dataset})
    dataset.save_to_disk("local_pdf_dataset")
    dataset.push_to_hub(dataset_repo_id)
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_repo_id}")


#############################################################################
#                     Summarization + LoRA Functions
#############################################################################

def preprocess_function(examples, tokenizer, body_key, summary_key, max_input_len=512, max_target_len=128):
    model_inputs = tokenizer(examples[body_key], truncation=True, max_length=max_input_len)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[summary_key], truncation=True, max_length=max_target_len)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_rouge_scores(
    model,
    dataset,
    tokenizer,
    device,
    body_key="body",
    summary_key="summary",
    max_length=128,
    num_beams=4
):
    """ Evaluate by generating from dataset[body_key], compare to dataset[summary_key]. """
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

def train_lora(
    base_model,
    dataset,
    tokenizer,
    model_repo_id,
    body_key="body",
    summary_key="summary",
    num_epochs=2
):
    """ Wrap base_model in LoRA, train on dataset => return final LoRA model. """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    base_model.to(device)

    from peft import LoraConfig, TaskType, get_peft_model
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q","v"]  # or q_proj/v_proj for T5
    )
    lora_model = get_peft_model(base_model, peft_config).to(device)

    def token_map_fn(examples):
        return preprocess_function(examples, tokenizer, body_key, summary_key)

    # Convert to tokenized form
    tokenized_ds = dataset.map(token_map_fn, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=lora_model, label_pad_token_id=-100)

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
        output_dir=f"{model_repo_id}_temp",
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

    final_eval = trainer.evaluate(tokenized_ds["test"])
    print("Trainer Evaluate (test set):", final_eval)
    return lora_model


#############################################################################
#                                  MAIN
#############################################################################

def main():
    pdf_folder = "sources"
    dataset_repo_id = "benitoals/my-pdf-dataset"

    model_name = "google/mt5-small"
    local_model_repo_id = "benitoals/my-lora-local"
    hf_model_repo_id    = "benitoals/my-lora-hf"
    combined_repo_id    = "benitoals/my-lora-combined"

    # Decide which external HF dataset to use:
    # For the smaller dataset => "CShorten/ML-ArXiv-Papers"
    # For the bigger => "armanc/scientific_papers" with config "arxiv"
    use_small = True
    if use_small:
        # small dataset
        huggingface_science_repo  = "CShorten/ML-ArXiv-Papers"
        huggingface_body_key      = "title"
        huggingface_summary_key   = "abstract"
        huggingface_config_name   = None  # doesn't have multiple configs
        # Because it only has one "train" split, we'll do a custom train/val/test
    else:
        huggingface_science_repo  = "armanc/scientific_papers"
        huggingface_body_key      = "article"
        huggingface_summary_key   = "abstract"
        huggingface_config_name   = "arxiv"

    # Step A) if local dataset not built, parse & push
    maybe_build_and_push_local_dataset(pdf_folder, dataset_repo_id)

    # Step B) load local dataset from HF
    local_data = load_dataset(dataset_repo_id)
    print("Local dataset loaded:", local_data)

    # Step 1) baseline => local test
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    base_model.to(device)

    baseline_rouge = get_rouge_scores(base_model, local_data["test"], tokenizer, device, "body", "summary")
    print("\n=== Baseline => local test ===")
    print(baseline_rouge)

    # Step 2) LoRA => local
    local_trained_model = train_lora(base_model, local_data, tokenizer, local_model_repo_id, "body", "summary", num_epochs=2)
    local_trained_model.eval()
    local_after_rouge = get_rouge_scores(local_trained_model, local_data["test"], tokenizer, device, "body", "summary")
    print("\n=== After LoRA on local => local test ===")
    print(local_after_rouge)

    # Step 3) Baseline model again => LoRA on external HF dataset => local test
    print(f"\n=== Fine-tune on HF dataset => {huggingface_science_repo} ===")
    baseline_model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    if use_small:
        # "CShorten/ML-ArXiv-Papers" has only a single "train" split
        # so we'll do scipapers = load_dataset(..., split="train")
        # then do our own random split
        scipapers_full = load_dataset(huggingface_science_repo, split="train")
        # scipapers_full => single Arrow dataset. We'll do a random split to train/validation/test
        # We'll create a list of dict from scipapers_full, then do split_train_val_test
        scipapers_list = list(scipapers_full)
        # If you want to subselect fewer for speed:
        # scipapers_list = scipapers_list[:2000]
        scipapers_dict = split_train_val_test(scipapers_list, 0.7, 0.15, 0.15)
        # We'll do huggingface_body_key = "title", huggingface_summary_key = "abstract"
        # But keep in mind that some rows might have missing values?
        # We can do a small filter or code a fallback
        # For now we assume there's a "title" and "abstract"
        scipapers_ds = scipapers_dict
    else:
        # "armanc/scientific_papers" with config "arxiv"
        if huggingface_config_name:
            scipapers_split = load_dataset(huggingface_science_repo, huggingface_config_name,
                                           split={"train": "train", "validation": "validation", "test": "test"},
                                           trust_remote_code=True)
        else:
            scipapers_split = load_dataset(huggingface_science_repo,
                                           split={"train":"train","validation":"validation","test":"test"},
                                           trust_remote_code=True)
        # optionally subselect
        # scipapers_split["train"] = scipapers_split["train"].select(range(2000))
        scipapers_ds = scipapers_split

    # Now train baseline_model2 on scipapers
    hf_trained_model = train_lora(baseline_model2,
                                  scipapers_ds,
                                  tokenizer,
                                  hf_model_repo_id,
                                  huggingface_body_key,
                                  huggingface_summary_key,
                                  num_epochs=1)
    hf_trained_model.eval()

    # Evaluate that model on local test
    hf_on_local_rouge = get_rouge_scores(hf_trained_model, local_data["test"], tokenizer, device, "body", "summary")
    print("\n=== HF model => local test ===")
    print(hf_on_local_rouge)

    # Step 4) from that HF-based model => train on local => local test
    final_model = train_lora(hf_trained_model, local_data, tokenizer, combined_repo_id, "body", "summary", num_epochs=2)
    final_model.eval()
    final_rouge = get_rouge_scores(final_model, local_data["test"], tokenizer, device, "body", "summary")
    print("\n=== HF + local => local test ===")
    print(final_rouge)

    # Print all four
    print("\n===== All four results =====")
    print("1) Baseline => local test          =>", baseline_rouge)
    print("2) LoRA local => local test        =>", local_after_rouge)
    print("3) LoRA HF => local test           =>", hf_on_local_rouge)
    print("4) HF + local => local test        =>", final_rouge)


if __name__ == "__main__":
    main()