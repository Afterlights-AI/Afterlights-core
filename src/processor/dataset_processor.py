import PyPDF2
import re
import csv

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    # Replace multiple spaces/newlines with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def extract_dialogues(text):
    dialogues = []
    pattern = re.compile(r'(PHILOSOPHER|YOUTH):\s*(.*?)(?=(PHILOSOPHER|YOUTH):|$)', re.DOTALL)
    for match in pattern.finditer(text):
        speaker = match.group(1)
        speech = match.group(2).strip()
        dialogues.append({"talker": speaker, "text": speech})
    return dialogues

def write_dialogues_to_csv(dialogues, csv_path):
    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["talker", "text"])
        writer.writeheader()
        for dialogue in dialogues:
            writer.writerow(dialogue)

pdf_text = read_pdf("dataset/the_courage_to_be_disliked.pdf")
cleaned_text = clean_text(pdf_text)
extracted_dialogues = extract_dialogues(cleaned_text)
write_dialogues_to_csv(extracted_dialogues, "dataset/the_courage_to_be_disliked.csv")  