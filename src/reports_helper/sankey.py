import os
import sys
from PyPDF2 import PdfReader, PdfWriter
import nltk
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the helper module
from helper import load_settings
nltk.download('stopwords')

def merge_pdfs(pdf_folder, output_path):
    pdf_writer = PdfWriter()
    pdf_titles = []
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_reader = PdfReader(pdf_path)
                for page_num in range(len(pdf_reader.pages)):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                pdf_titles.append(file)
    with open(output_path, 'wb') as out_pdf:
        pdf_writer.write(out_pdf)
    return pdf_titles

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def main():
    settings = load_settings()
    path = settings.get('reports_path')
    pdf_folder = f'{path}/reports'
    report_pdf = f'{path}/analysis_report_BPIC_2019.pdf'
    merged_pdf_path = 'merged.pdf'

    # Step 1: Merge PDFs and get titles
    pdf_titles = merge_pdfs(pdf_folder, merged_pdf_path)

    # Step 2: Extract text from each PDF
    pdf_texts = [extract_text_from_pdf(os.path.join(pdf_folder, title)) for title in pdf_titles]
    report_text = extract_text_from_pdf(report_pdf)

    # Step 3: Define compound phrases to search for
    compound_phrases = [
        "Vinyl Acetate Ethylene", "Process Automation & Instrumentation",
        "Intermediate Bulk Containers", "Pure Resins & Pigments",
        "MR0 (Components)", "Facility Management", "Office Supplies", "Sales",
        "DRUM", "Digital Marketing", "Road Packed", "Logistics",
        "Marketing", "Standard PO", "EC Purchase order", "Variability", "Sea", "Warehousing",
        "Top Performers", "Benchmark", "Score", "Bottleneck", "Consigment", "NPR",
        "Distribution", "Marketing", "Packaging",  "Sequence", "Sales"
    ]

    # Step 4: Map phrases to individual PDFs
    phrase_pdf_mapping = {phrase: {title: 0 for title in pdf_titles} for phrase in compound_phrases}
    for title, text in zip(pdf_titles, pdf_texts):
        for phrase in compound_phrases:
            phrase_pdf_mapping[phrase][title] = text.lower().count(phrase.lower())

    # Step 5: Create Sankey Diagram
    sources = []
    targets = []
    values = []
    label_list = list(pdf_titles) + compound_phrases
    for phrase, pdf_map in phrase_pdf_mapping.items():
        for pdf, count in pdf_map.items():
            if count > 0:
                sources.append(label_list.index(pdf))
                targets.append(label_list.index(phrase))
                values.append(count)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_list,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        ))])

    fig.update_layout(title_text="Sankey Diagram of Compound Phrases in PDF Documents", font_size=10)
    fig.show()

if __name__ == '__main__':
    main()
