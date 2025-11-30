import json
import pandas as pd
from io import StringIO
from google import genai
import os
api_key = 'AIzaSyBmAvLqApnVms-eRRSAWyZ9Wth_9NC_hlc'
client = genai.Client(api_key=api_key)

data = pd.read_csv('./banking77/train.csv')
# create dir auxillary if not exists
if not os.path.exists('./auxillary'):
    os.makedirs('./auxillary')
classes = data['category'].unique().tolist()
T = 60
task_cnt = 0
task_dataframe = pd.DataFrame(columns=['task_name', 'task_description'])

for i in range(T // 3):
    # sample 5 of the classes into a dict, with 3 examples each
    sampled_train_classes = {}
    categories_sample = pd.Series(classes).sample(5, random_state=42).tolist()
    for category in categories_sample:
        sampled_train_classes[category] = data[data['category'] == category]['text'].sample(3).tolist()

    sampled_classes = {}
    categories_sample = pd.Series(classes).sample(3, random_state=42).tolist()
    for category in categories_sample:
        sampled_classes[category] = data[data['category'] == category]['text'].sample(3, random_state=42).tolist()

    prompt = "Come up with 3 helpful auxillary tasks, for training an LLM to classify customer service queries into the following categories:\n\n"
    for category, examples in sampled_classes.items():
        prompt += f"Label: {category}\n"
        prompt += "Example texts from customers:\n"
        for example in examples:
            prompt += f"- {example}\n"
        prompt += "\n"

    prompt += "Provide 3 auxillary tasks that would help an LLM learn to classify these queries effectively. All auxillary tasks must only have exactly 1 label (a word, class or a string). Each task needs to be of format {text: str, label: str}. Here are current auxillary tasks available:\n"
    for category, examples in sampled_train_classes.items():
        prompt += f"Label: {category}\n"
        prompt += "Example texts from customers:\n"
        for example in examples:
            prompt += f"- {example}\n"
        prompt += "\n"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    for i in range(1, 4):
        task_dict = {}
        task_prompt = prompt
        task_prompt += response.text
        task_prompt += f"\nNow, for the auxillary task number {i} generated above, generate task description."
        task_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=task_prompt,
        )
        task_name = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Generate a concise name for the following auxillary task:\n\n{task_response.text}\n\nOutput only the task name, it has to be usable as file name.",
        )
        task_description = task_response.text
        task_dataframe = pd.concat([task_dataframe, pd.DataFrame({'task_name': [task_name.text + str(task_cnt)], 'task_description': [task_description]})], ignore_index=True)
        task_prompt += task_response.text
        task_prompt += f"\nNow, for the auxillary task number {i} generated above, generate 5 training data points (query + label) that would help a model learn to classify customer service queries effectively."
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
            df.to_csv(f'./auxillary/{task_name.text + str(task_cnt)}.csv', index=False)
        except Exception as e:
            print(f"Error saving file for task {task_name.text + str(task_cnt)}: {e}")

        task_cnt += 1

task_dataframe.to_csv('./auxillary/task_descriptions.csv', index=False, sep=';')
