import openai
import requests
import base64

class OpenAIImageAnalyzer:
    def __init__(self, api_key = None, image_path=None, prompt='Analyze the image'):
        if not api_key:
            self.api_key = "XXXX"
        else:
            self.api_key =  api_key
        self.image_path = image_path
        self.prompt = prompt
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def encode_image(self):
        with open(self.image_path, "rb") as img_file:
            base64_encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return base64_encoded
    
    def analyze_image(self):
        base64_image = self.encode_image()
        response = self.client.chat.completions.create(
    model='gpt-4-vision-preview',
    messages=[
        {
            'role': 'user',
            'content': [
                    {'type': 'text', 'text': self.prompt},
                    {'type': 'image_url', 
                     'image_url': {
                         'url': f'data:image/png;base64,{base64_image}'
                         },
                        }, 
                ],
        }
    ],
    max_tokens=700,
)

        return response.choices[0].message.content
    
def analyze_image(api_key, image_path,prompt='Analyze the image'):
    analyzer = OpenAIImageAnalyzer(api_key,image_path,prompt=prompt)
    response = analyzer.analyze_image()
    return response
    

# Example usage
if __name__ == '__main__':
    api_key = None
    category = 'cat_dim_5'
    image_path = f'/Users/urszulajessen/code/gitHub/WISE/data/results/data_BPIC_2019/{category}_adjusted_boxplot.png'
    prompt = """
    You are experienced data analyst. This data shows the performance of a process scored for different process relevant features.
    The data comes from the BPIC 2019 dataset. Analyze the image and provide insights. Create the plan for next steps.
    Make it consise and actionable.
    """
    response = analyze_image(api_key,image_path,prompt=prompt)
    print(response)

