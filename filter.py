import pandas as pd
import random

from google import genai
api_key = 'AIzaSyBmAvLqApnVms-eRRSAWyZ9Wth_9NC_hlc'
client = genai.Client(api_key=api_key)

aux_data = pd.read_csv('./auxillary/task_descriptions.csv', delimiter=';')
# add columns score and used to aux_data
aux_data['score'] = 0
aux_data['used'] = 0
train_data = pd.read_csv('./banking77/train.csv')
T = aux_data.shape[0]

for j in range(T):
    sample_weights = aux_data['used'].apply(lambda x: 1 / (1 + x))
    sample_weights = sample_weights / sample_weights.sum()
    aux_data_sample = aux_data.sample(n=5, weights=sample_weights, random_state=j)
    prompt = ""
    for index, row in aux_data_sample.iterrows():
        prompt += f"Task Name: {row['task_name']}\n"
        prompt += f"Task Description: {row['task_description']}\n\n"
        task_data = pd.read_csv(f'./auxillary/{row["task_name"]}.csv')
        for i, task_row in task_data.iterrows():
            prompt += f"question: {task_row['text']}, answer: {task_row['label']}\n"

    sample_classes = random.sample(train_data['category'].unique().tolist(), 5)
    sample_questions = train_data[train_data['category'].isin(sample_classes)].sample(15, random_state=42)
    prompt += f"Task Name: Classification into 5 classes: {', '.join(sample_classes)}\n"
    for i, row in sample_questions.iterrows():
        prompt += f"question: {row['text']}, answer:\n"
    prompt += "\nProvide correct answers, separated by newlines for the above questions based on the task descriptions provided. Give only the answers, no explanations.\n"

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    list_of_answers = response.text.strip().split('\n')
    accuracy = 0
    cnt = 0
    for i, row in sample_questions.iterrows():
        if row['category'] == list_of_answers[cnt]:
            accuracy += 1
        cnt += 1
    print(f"Accuracy: {accuracy}/{len(sample_questions)}")

    for index, row in aux_data_sample.iterrows():
        aux_data.loc[aux_data['task_name'] == row['task_name'], 'used'] += 1
        aux_data.loc[aux_data['task_name'] == row['task_name'], 'score'] += accuracy
aux_data['score'] = aux_data['score'] / aux_data['used'].replace(0, 1)
aux_data.to_csv('./auxillary/task_descriptions.csv', index=False, sep=';')

# filter only 50% of auxillary tasks based on score
threshold = aux_data['score'].median()
filtered_aux_data = aux_data[aux_data['score'] >= threshold]
filtered_aux_data.to_csv('./auxillary/filtered_task_descriptions.csv', index=False, sep=';')