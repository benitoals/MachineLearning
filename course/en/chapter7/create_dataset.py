import os
import re
import csv
import PyPDF2  # pip install PyPDF2
from datasets import Dataset  # pip install datasets

def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from a PDF using PyPDF2.
    Returns a single string with the entire text.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            # If extract_text() returns None, use empty string
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def split_abstract_body(raw_text):
    """
    Given the PDF's raw text, attempt to extract:
       - abstract (as summary)
       - main body
    The 'title' is no longer extracted here, because we'll
    use the filename as the 'title' in our CSV.
    """
    lines = raw_text.splitlines()
    # Remove empty lines and extra spaces
    lines = [ln.strip() for ln in lines if ln.strip()]

    abstract_lines = []
    body_lines = []
    inside_abstract = False

    # Regex to catch variations of "Abstract"
    abstract_pattern = re.compile(r"^\s*Abstract\b", re.IGNORECASE)

    for line in lines:
        if not inside_abstract:
            # Haven't encountered the abstract section yet
            if abstract_pattern.match(line):
                inside_abstract = True
                # If the line is exactly "Abstract", skip it.
                if line.lower().strip() == "abstract":
                    continue
                else:
                    abstract_lines.append(line)
            else:
                body_lines.append(line)
        else:
            # Inside the abstract.
            # When a new section starts, assume abstract ended.
            if re.match(r"^\s*(keywords|introduction|1\.|2\.|I\.)\b", line, re.IGNORECASE):
                inside_abstract = False
                body_lines.append(line)
            else:
                abstract_lines.append(line)

    abstract = " ".join(abstract_lines).strip()
    body = " ".join(body_lines).strip()
    return abstract or "", body or ""

def create_dataset_from_pdfs(pdf_folder, output_folder="my_pdf_dataset", output_csv="my_pdf_data.csv"):
    """
    Reads all PDFs from `pdf_folder`, extracts the (filename, title=filename, abstract, body),
    and saves them in a CSV in `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    output_csv_path = os.path.join(output_folder, output_csv)

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write CSV header
        writer.writerow(["filename", "title", "abstract", "body"])

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            raw_text = extract_text_from_pdf(pdf_path)

            abstract, body = split_abstract_body(raw_text)
            title = pdf_file  # Use the PDF filename as the title

            # Fallback to an empty string if any field is falsy.
            writer.writerow([pdf_file or "", title or "", abstract or "", body or ""])

    print(f"Done! CSV dataset saved to: {output_csv_path}")

def load_dataset_from_csv(csv_path):
    """
    Loads the CSV file into a list of dictionaries.
    """
    examples = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            examples.append({
                "filename": row.get("filename", ""),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "body": row.get("body", "")
            })
    return examples

def clean_data(data):
    """
    Recursively traverse data (dictionary or list) and replace any Python ellipsis (i.e., `...`)
    with an empty string.
    """
    if isinstance(data, dict):
        return {key: clean_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    else:
        return "" if data is Ellipsis else data

if __name__ == "__main__":
    pdf_input_folder = "sources"  # Folder containing your source PDFs
    
    # Create the CSV dataset from PDFs
    create_dataset_from_pdfs(pdf_input_folder)
    
    # Load the dataset from the created CSV file
    csv_path = os.path.join("my_pdf_dataset", "my_pdf_data.csv")
    examples = load_dataset_from_csv(csv_path)
    
    # Clean the examples in case any value is the Python ellipsis
    cleaned_examples = clean_data(examples)
    
    # Convert the list of dictionaries into a HuggingFace Dataset
    dataset = Dataset.from_list(cleaned_examples)
    print("Dataset created successfully:")
    print(dataset)