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
from code_old.instructions_all import instruction_4evaluation_0norm, instruction_4evaluation_norm, instruction_4evaluation_conflict_norm, instruction_4evaluation_2norms


model_name = 'gpt-3.5-turbo' # llama3-70b-instruct, llama3-8b-instruct, gpt-3.5-turbo, gpt-4o
task_setting = '2norms' # 0norm, norm, conflict-norm,2norms
dataset = 'moral_conflicts' # original_moral_stories, moral_conflicts

if task_setting == '0norm':
    instruction = instruction_4evaluation_0norm
elif task_setting == 'norm':
    instruction = instruction_4evaluation_norm
elif task_setting == 'conflict-norm':
    instruction = instruction_4evaluation_conflict_norm
else:
    instruction = instruction_4evaluation_2norms


testing_data_path = f'../datasets/new_datasets/final_datasets/{dataset}.json' # final data
with open(testing_data_path, 'r') as f:
    testing_data = json.load(f)

output_dir = f'../exps/{model_name}/'
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{dataset}_{task_setting}.json"


if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        results = json.load(f)
    print(f"已经存在{len(results)}条数据,从上次断点继续")
else:
    results = []

exist_num = len(results)
for item in tqdm(testing_data[exist_num:]):
    choice = random.choice([0,1])

    if choice == 0: # A为正确选项
        instruction_ = instruction
        item['Truth'] = 'A'
    elif choice == 1 and task_setting!='2norms': # B为正确选项
        item['Truth'] = 'B'
        instruction_ = instruction.replace("A.{Moral_action}\nB.{Immoral_action}","A.{Immoral_action}\nB.{Moral_action}")
    else: # 2norms实验，既交换选项A和B，又交换norm与conflict-norm的位置
        item['Truth'] = 'B'
        instruction_ = instruction.replace("Moral Norm 1:{Norm}\nMoral Norm 2:{Conflict-norm}", "Moral Norm 1:{Conflict-norm}\nMoral Norm 2:{Norm}")
        instruction_ = instruction.replace("A.{Moral_action}\nB.{Immoral_action}","A.{Immoral_action}\nB.{Moral_action}")

    prompt = get_prompt(instruction_, item)
    item['evaluation_prompt'] = prompt
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
        item['model_answer'] = Answer
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
        time.sleep(5)
        item['model_answer'] = Answer
    results.append(item)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

precision, recall, f1 = cal_metrics_4_evaluation(results)
print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


