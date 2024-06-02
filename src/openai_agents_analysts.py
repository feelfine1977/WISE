import os
import openai
import base64
import requests
from fpdf import FPDF, XPos, YPos

class OpenAIImageAnalyzer:
    def __init__(self, api_key=None, image_paths=None, prompt='Analyze the images'):
        self.api_key = api_key or "XXXX"
        self.image_paths = image_paths if image_paths is not None else []
        self.prompt = prompt
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.responses = []  # Store responses for each image

    def encode_images(self, image_path):
        with open(image_path, "rb") as img_file:
            base64_encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return base64_encoded



    def analyze_image_4o(self, image_paths):
        responses = []
        for image_path in image_paths:
            base64_image = self.encode_images(image_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            #print(response.json())
            responses.append(response.json()['choices'][0]['message']['content'])
        return "\n\n".join(responses)

    def analyze_summary(self):
        messages = [{'role': 'user', 'content': self.prompt}]
        for response in self.responses:
            messages.append({'role': 'user', 'content': response})

        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            max_tokens=4096,
        )

        return response.choices[0].message.content


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add a Unicode font (e.g., DejaVu)
        self.add_font('DejaVu', '', '/Users/urszulajessen/code/gitHub/WISE/data/fonts/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', '/Users/urszulajessen/code/gitHub/WISE/data/fonts/DejaVuSans-Bold.ttf', uni=True)

    def header(self):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, 'BPIC 2019 Data Analysis Report', new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.RIGHT, new_y=YPos.NEXT)
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('DejaVu', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def chapter_image(self, image_path, width=190):
        self.image(image_path, w=width)
        self.ln(10)

    def full_page_title(self, title):
        self.add_page()
        self.set_font('DejaVu', 'B', 36)
        self.cell(0, 200, title, 0, 1, 'C')


class ReportGenerator:
    def __init__(self, api_key, folder_path, categories=None):
        self.api_key = api_key
        self.folder_path = folder_path
        if categories:
            self.categories = categories
        else:
            self.categories = self.get_categories()
        self.responses = []

    def get_categories(self):
        if self.categories is None:
            categories = set()
            categories = ['cat_dim_2', 'cat_dim_3',  'cat_dim_5', 'cat_dim_6', 'cat_dim_7', 'cat_dim_8', 'cat_dim_9', 'cat_dim_10', 'cat_dim_12']
        else:
            categories = set(self.categories)
        return list(categories)

    def analyze_category(self, category):
        image_paths = [
            os.path.join(self.folder_path, f"{category}_adjusted_boxplot.png"),
            os.path.join(self.folder_path, f"{category}_heatmap.png")
        ]
        prompt = f"""
        You are an experienced data analyst. This data shows the performance of a process scored for different process relevant features.
        The data comes from the BPIC 2019 dataset. Analyze the images and provide insights. Create the plan for next steps.
        The meaning of columns for heatmap- columns show score between 0 and 1. The higher the score the better the process performance.:
            'score_found_case_id' - score is given for presence of mandatory activities in the process, 
            'score_sequential_case_id' - score is given for correct order of activities in the process, 
            'score_equilibrium_case_id' - score is given for balanced distribution of activities in the process, 
            'score_singular_case_id' - score is given for absence of duplicate activities in the process, 
            'score_exclusion_case_id' - score is given for absence of manual, costly or otherwise not wanted activities in the process.
        Give clear categories and dimensions names if possible, use numbers to quantify the performance.
        Make it concise and actionable. Put the main points instead full sentences.
        """
        analyzer = OpenAIImageAnalyzer(api_key=self.api_key, image_paths=image_paths, prompt=prompt)
        response = analyzer.analyze_image_4o(image_paths)
        self.responses.append((category, response, image_paths))

    def generate_summary(self):
        partial_results = "\n\n".join([response for _, response, _ in self.responses])
        summary_prompt = f"""
        You are an experienced process analyst specializing in data analysis and process improvement. 
        Based on the following partial results, prepare an insightful 
        summary about the main problems in the analyzed process and the areas for improvement. 
        Create a step-by-step plan for Project Improvement:

        {partial_results}

        Provide a detailed and actionable plan. Use numbers to quantify the performance. Make it concise and actionable. 
        Put the main points instead full sentences.
        Use full categories names and dimensions names if possible, so it is clear what have to be done.
        """
        analyzer = OpenAIImageAnalyzer(api_key=self.api_key, prompt=summary_prompt)
        summary_response = analyzer.analyze_summary()  
        return summary_response

    def generate_report(self, output_path):
            pdf = PDF()
            pdf.add_page()

            summary = self.generate_summary()
            pdf.full_page_title("MANAGEMENT SUMMARY")
            pdf.chapter_body(summary)

            pdf.full_page_title("DETAILS OF ANALYSIS")

            for category, response, image_paths in self.responses:
                pdf.add_page()
                pdf.chapter_title(f"Category: {category}")
                pdf.chapter_body(response)
                for image_path in image_paths:
                    pdf.chapter_image(image_path)

            pdf.output(output_path)
            return summary

    def run_analysis(self, output_path, summary=False):
        for category in self.categories:
            self.analyze_category(category)

        summary_text = self.generate_report(output_path)

        if summary:
            with open(output_path.replace('.pdf', '.txt'), 'w') as summary_file:
                summary_file.write(summary_text)
            return summary_text
        
def main(summary_needed=True,categories=None):
    api_key = "YOUR_API_KEY"  
    folder_path = '/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/'
    output_path = '/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/analysis_report_BPIC_2019.pdf'

    report_generator = ReportGenerator(api_key, folder_path, categories=categories)
    summary_text = report_generator.run_analysis(output_path, summary=summary_needed)
    if summary_needed:
        return summary_text

if __name__ == '__main__':
    api_key = "YOUR_API_KEY"  
    folder_path = '/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/'
    output_path = '/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/analysis_report_BPIC_2019.pdf'

    report_generator = ReportGenerator(api_key, folder_path)
    summary_needed = True  # Change to False if full report is needed
    summary_text = report_generator.run_analysis(output_path, summary=summary_needed)
    if summary_needed:
        print(summary_text)

