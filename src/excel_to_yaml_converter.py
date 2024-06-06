import pandas as pd
import yaml
from helper import load_settings

class ExcelToYAMLConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {'Layers': []}

    def read_excel_sheet(self, sheet_name):
        return pd.read_excel(self.file_path, sheet_name=sheet_name)

    def add_layer(self, layer_name, categories):
        self.data['Layers'].append({'Layer': layer_name, 'Categories': categories})

    def generate_category(self, category_name, score_types):
        category = {'Category': category_name}
        for score_type, sheet_desc in score_types.items():
            #add score_class score_desc
            score_class = sheet_desc[1]
            sheet_name = sheet_desc[0]
            df = self.read_excel_sheet(sheet_name)
            if 'Sequential' in score_type:  # Special handling for sequential scores
                events = [[row[0], row[1]] for index, row in df.iterrows()]  # Pair preceding and follower activities as lists
                weights = df['Weight'].tolist()  # Assuming weights are in a column named "Weight"
            else:
                events = df.iloc[:, 0].tolist()  # Assuming the events are in the first column
                weights = df.iloc[:, 1].tolist()  # Assuming weights are in the second column
            category[score_type] = {'events': events, 'weights': weights,'score_class': score_class}
        return [category]

    def save_to_yaml(self, output_path):
        with open(output_path, 'w') as file:
            yaml.safe_dump(self.data, file, default_flow_style=False, allow_unicode=True)

                   
if __name__ == "__main__":
    settings = load_settings()
    path_base = settings.get('data_path')
    path = f"{path_base}data_BPIC_2019//WISE_Framework_BPIC_2019.xlsx"
    converter = ExcelToYAMLConverter(path)
    category_standard = converter.generate_category('Standard', {
    'Foundational_Scores': ('Foundational Scores', 'found'),
    'Sequential_Scores':('Sequential Scores', 'sequnt'),
    'Equilibrium_Scores': ('Equilibrium Scores','equilib'),
    'Singular_Scores': ('Singular Scores', 'sing'),
    'Exclusion_Scores': ('Exclusion Scores', 'exclusion'),
    'Change_Scores' : ('Change Scores', 'exclusion'),
})
    converter.add_layer('General_Process_Standards', category_standard)
    converter.save_to_yaml(f'{path_base}data_BPIC_2019/BPIC_2019_weights.yaml')