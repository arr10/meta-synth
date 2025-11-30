import pandas as pd
import random
from io import StringIO
from google import genai
import os
api_key = 'AIzaSyBmAvLqApnVms-eRRSAWyZ9Wth_9NC_hlc'
client = genai.Client(api_key=api_key)

aux_data = pd.read_csv('./auxillary/filtered_task_descriptions.csv', delimiter=';')

for i, row in aux_data.iterrows():
        task_dict = {}
        task_prompt = f"Task Name: {row['task_name']}\n"
        task_prompt += f"Task Description: {row['task_description']}\n\n"
        task_prompt += "Generate 100 training data points (query + label) that would help a model learn to classify customer service queries effectively."
        task_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=task_prompt,
        )

        response_csv = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Convert the following training data points into a csv format with 'text' and 'label' columns:\n\n{task_response.text}\n\nOutput only the csv data. Use semicolon as delimiter. Do not use any other delimiter. Do not use semicolon in the text or label values.",
        )
        # convert csv string to dataframe
        try:
            if response_csv.text.startswith("```csv"):
                csv_data_sliced = response_csv.text.strip()[6:-3].strip()
            else:
                csv_data_sliced = response_csv.text.strip() 
            df = pd.read_csv(StringIO(csv_data_sliced), delimiter=';')
            df.to_csv(f'./auxillary_full/{row['task_name']}.csv', index=False)
        except Exception as e:
            print(f"Error saving file for task {row['task_name']}: {e}")
