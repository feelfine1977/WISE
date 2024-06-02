import os
from PyPDF2 import PdfReader, PdfWriter
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def find_common_words(text1, text2):
    words1 = remove_stopwords_and_common_words(text1)
    words2 = remove_stopwords_and_common_words(text2)
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    common_words = counter1 & counter2
    return common_words

def main():
    pdf_folder = '/Users/urszulajessen/code/gitHub/WISE/src/reports_helper/reports'
    report_pdf = '/Users/urszulajessen/code/gitHub/WISE/src/reports_helper/analysis_report_BPIC_2019.pdf'
    merged_pdf_path = 'merged.pdf'

    # Step 1: Merge PDFs
    pdf_titles = merge_pdfs(pdf_folder, merged_pdf_path)

    # Step 2: Extract text from merged PDF and report PDF
    merged_text = extract_text_from_pdf(merged_pdf_path)
    report_text = extract_text_from_pdf(report_pdf)

    # Step 3: Find common words between the texts
    common_words = find_common_words(merged_text, report_text)

    # Step 4: Delete words that are common and not significant
    list_delete = ['process','activity','activities','case','cases','event','events','time','timestamp', \
                   'several','5', 'categories', 'data', '1','2','3','4','6','7','8','9','10','show', 'around', 'focusing',
                   'actions', '0', 'set', 'specific', 'adress', 'absence', 'others','cause', 'address', 'report', 'areas',
                   'focus','across','regular','identify','investigate','review','root', 'improve','like','recommendations',
                   'high', 'category', 'overall', 'next', 'detailed', 'best', 'presence', 'correct', 'indicating','reduce',
                   'ensure', 'highest', 'bpic', 'consistent', 'understand', 'analysis', '2019', 'low', 'high', 'mandatory']
    for word in list_delete:
        common_words.pop(word, None)

    # Step 5: Sort the results and output the results, show only top 50
    common_words = dict(sorted(common_words.items(), key=lambda x: x[1], reverse=True))
    print("Common words and their frequencies:")
    for word, freq in list(common_words.items())[:50]:
        print(f"{word}: {freq}")

    # Bar chart for word distribution
    common_words_list = list(common_words.items())[:50]
    words, freqs = zip(*common_words_list)
    
    plt.figure(figsize=(12, 8))
    plt.bar(words, freqs)
    plt.xticks(rotation=90)
    plt.title('Top 50 Common Words and Their Frequencies')
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.tight_layout()
    plt.savefig('common_words_distribution.png')
    plt.show()

    # Heatmap for word distribution across documents
    word_counts = {title: Counter(remove_stopwords_and_common_words(extract_text_from_pdf(os.path.join(pdf_folder, title)))) for title in pdf_titles}
    word_counts_df = pd.DataFrame(word_counts).fillna(0)
    
    # Only include shared words and get top 30 words
    shared_words_df = word_counts_df.loc[word_counts_df.index.intersection(common_words.keys())]
    top_shared_words = shared_words_df.sum(axis=1).sort_values(ascending=False).head(30).index
    shared_words_df = shared_words_df.loc[top_shared_words]

    # Get top 10 PDFs with the most shared words
    top_pdfs = shared_words_df.sum(axis=0).sort_values(ascending=False).head(10).index
    shared_words_df = shared_words_df[top_pdfs]

    plt.figure(figsize=(16, 12))
    sns.heatmap(shared_words_df, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Heatmap of Top 30 Shared Words in Top 10 Documents')
    plt.xlabel('Documents')
    plt.ylabel('Words')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('shared_words_heatmap.png')
    plt.show()

if __name__ == '__main__':
    main()
