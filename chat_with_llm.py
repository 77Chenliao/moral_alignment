import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = "sk-7bd6acf8d60c46369a6c89c26c714393"
import json
import time
import os
from openai import OpenAI
from tqdm import tqdm
api_key = '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'    # Fdu
from sklearn.metrics import precision_score, recall_score



data_path = r'xx'
with open(data_path, 'r') as f:
    data = json.load(f)


model_name = 'qwen-turbo'
instruction = 'xx'

new_data = []
for item in tqdm(data):
    prompt = 'xx'
    messages = [{'role': Role.SYSTEM, 'content': "xx"},
                {'role': Role.USER, 'content': f'{prompt}'}]
    if 'gpt' in model_name:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        Answer = completion.choices[0].message.content
    else:
        completion = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        )
        Answer = completion.output.choices[0]['message']['content']
        time.sleep(2)