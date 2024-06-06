import os
import sys
from PyPDF2 import PdfReader, PdfWriter
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

def remove_stopwords_and_common_words(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

def find_common_words(texts):
    combined_counter = Counter()
    for text in texts:
        words = remove_stopwords_and_common_words(text)
        combined_counter.update(words)
    return combined_counter

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

    # Step 3: Find common words across all PDFs
    common_words = find_common_words(pdf_texts)
    
    # Step 4: Remove non-significant words
    list_delete = ['process', 'activity', 'activities', 'case', 'cases', 'event', 'events', 'time', 'timestamp',
                   'several', '5', 'categories', 'data', '1', '2', '3', '4', '6', '7', '8', '9', '10', 'show', 'around', 'focusing',
                   'actions', '0', 'set', 'specific', 'adress', 'absence', 'others', 'cause', 'address', 'report', 'areas',
                   'focus', 'across', 'regular', 'identify', 'investigate', 'review', 'root', 'improve', 'like', 'recommendations',
                   'high', 'category', 'overall', 'next', 'detailed', 'best', 'presence', 'correct', 'indicating', 'reduce',
                   'ensure', 'highest', 'bpic', 'consistent', 'understand', 'analysis', '2019', 'low', 'high', 'mandatory',
                   'higher', 'key', 'conduct', 'good', 'lower', 'improvements', 'match', 'based', 'causes', 'steps', 'processes',
                   'improvement', 'insights', 'dimensions'
                   ]
    for word in list_delete:
        common_words.pop(word, None)

    # Step 5: Get the top 50 common words
    top_common_words = dict(common_words.most_common(100))

    # Step 6: Map words to individual PDFs
    word_pdf_mapping = {word: {title: 0 for title in pdf_titles} for word in top_common_words.keys()}
    for title, text in zip(pdf_titles, pdf_texts):
        words = remove_stopwords_and_common_words(text)
        word_count = Counter(words)
        for word in top_common_words.keys():
            if word in word_count:
                word_pdf_mapping[word][title] = word_count[word]

    # Step 7: Create Sankey Diagram
    sources = []
    targets = []
    values = []
    label_list = list(pdf_titles) + list(top_common_words.keys())
    for word, pdf_map in word_pdf_mapping.items():
        for pdf, count in pdf_map.items():
            if count > 0:
                sources.append(label_list.index(pdf))
                targets.append(label_list.index(word))
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

    fig.update_layout(title_text="Sankey Diagram of Top 50 Common Words in PDF Documents", font_size=10)
    fig.show()

if __name__ == '__main__':
    main()
