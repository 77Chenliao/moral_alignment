"""
    利用生成的数据集对大模型进行测试
"""
import dashscope
from dashscope.api_entities.dashscope_response import Role, Message
import json
import time
from openai import OpenAI
from tqdm import tqdm
import random
random.seed(2024)
import os
from utils import get_prompt, ali_api_key, openai_api_key, cal_metrics_4_evaluation
from instructions import instruction_4_moral_action_choice_basic, instruction_4_moral_action_choice_conflict_norm, instruction_4_moral_action_choice_norm, instruction_4_moral_action_choice_2norms



data_path = f'./moral_conflicts.json' # final data
with open(data_path, 'r',encoding='utf-8') as f:
    testing_data = json.load(f)

model_name = 'gpt-3.5-turbo' # llama3-8b-instruct, gpt-3.5-turbo, gpt-4o
task_setting = '2norms' # basic, conflict-norm, norm, 2norms

if task_setting == 'basic':
    instruction = instruction_4_moral_action_choice_basic
elif task_setting == 'norm':
    instruction = instruction_4_moral_action_choice_norm
elif task_setting == 'conflict-norm':
    instruction = instruction_4_moral_action_choice_conflict_norm
elif task_setting == '2norms':
    instruction = instruction_4_moral_action_choice_2norms
else:
    instruction = instruction_4_moral_action_choice_basic


output_dir = f'../exps/{model_name}/'
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/moral_action_choice_{task_setting}.json"


if os.path.exists(output_path):
    with open(output_path, 'r',encoding='utf-8') as f:
        results = json.load(f)
    print(f"已经存在{len(results)}条数据,从上次断点继续")
else:
    results = []

exist_num = len(results)
for item in tqdm(testing_data[exist_num:]):
    choice = random.choice([0,1])
    if choice == 0: # A为正确选项
        item['truth'] = 'A'
        instruction_ = instruction
    elif choice == 1: # B为正确选项
        item['truth'] = 'B'
        instruction_ = instruction.replace("A.{Action}\nB.{Conflict_action}","A.{Conflict_action}\nB.{Action}")
    prompt = get_prompt(instruction_, item)
    if 'gpt' in model_name:
        messages = [{'role': Role.SYSTEM, 'content': "You're a moral AI assistant."},
                    {'role': Role.USER, 'content': f'{prompt}'}]
        client = OpenAI(api_key=openai_api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1e-5
        )
        Answer = completion.choices[0].message.content
        item['result'] = Answer
    else:
        messages = [Message(role=Role.SYSTEM, content="You are a moral AI assistant."),
                    Message(role=Role.USER, content=prompt)]
        completion = dashscope.Generation.call(
        model=model_name,
        api_key=ali_api_key,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        temperature=1e-5
        )
        Answer = completion.output.choices[0]['message']['content']
        time.sleep(7)
        item['result'] = Answer
    results.append(item)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

precision, recall, f1 = cal_metrics_4_evaluation(results)
print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


