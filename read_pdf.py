import pypdf
import sys

try:
    reader = pypdf.PdfReader('Debyez AI intern Assessment Steps.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open('assessment_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("PDF text extracted to assessment_text.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
