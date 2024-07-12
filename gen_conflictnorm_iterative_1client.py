import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = "sk-7bd6acf8d60c46369a6c89c26c714393"
import json
from openai import OpenAI
from tqdm import tqdm
from utils import extract_norm, get_prompt, extract_situation
api_key = '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'    # Fdu
import os
import  sys
from datetime import datetime
from instructions_all import instruction_4gen_confnorm_4oneclient, instruction_4rec_confnorm_4oneclient, instruction_4rec_situation_4oneclient, instruction_4gen_situation_4oneclient

from judge_4_situation import judge_4_situation
from judge_4_conflict_norm import judge_4_conflict_norm

input_dir = "../datasets/moral_stories"
dataset_name = 'test_demo'
data_path = f"{input_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

model_name = 'gpt-4o-2024-05-13'
client = OpenAI(api_key=api_key)

prompt_strategy = 'generated_iteratively_1client'
output_dir = f"../datasets/new_datasets/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{dataset_name}_{prompt_strategy}_gpt-4o.json"
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
else:
    new_data = []
output_path_4_messages_history = f"{output_dir}/{dataset_name}_{prompt_strategy}_gpt-4o_messages_history_all.json"
if os.path.exists(output_path_4_messages_history):
    with open(output_path_4_messages_history, 'r') as f:
        messages_history_all = json.load(f)
    print(f"已经存在{len(messages_history_all)}条数据,从上次断点继续")
else:
    messages_history_all = []


MAX_ITER = 3
Temperature_4_conflict_norm = 0.01
Temperature_4_situation = 1.2


existing_data_len = len(new_data)
# for item in tqdm(data[existing_data_len:]):
for index, item in enumerate(data[:1]):
    messages_history = []
    judgement_history = []
    new_conflict_norm = ''
    new_situation = ''
    iter_count = 1
    print(f"第{existing_data_len+index+1}/{len(data)}条数据开始生成")
    while True:
        if iter_count > MAX_ITER:
            index += 1
            break
        print(f"  第{iter_count }次迭代")
        if  iter_count == 1:
            # 首先生成conflict-norm
            instruction = instruction_4gen_confnorm_4oneclient
            prompt = get_prompt(instruction, item)
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_conflict_norm
            )
            Answer = completion.choices[0].message.content
            print(f"    conflict-norm完成")
            # 将生成的conflict-norm加入历史
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            item['conflict-norm'] = new_conflict_norm
            conflict_norm_judgement = judge_4_conflict_norm(new_conflict_norm)
            print(f"    conflict-norm_judgement:{conflict_norm_judgement}")
            # 然后扩充situation
            instruction = instruction_4gen_situation_4oneclient
            original_length = len(f"{item['situation']} {item['intention']}".split())
            prompt = get_prompt(instruction, item)
            prompt = prompt.replace('{length_limit}', f"{max(100,2*original_length)}")
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_situation
            )
            print(f"    situation完成")
            Answer = completion.choices[0].message.content
            # 将生成的situation加入历史
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
            response_list, situation_judgement = judge_4_situation({'situation': new_situation,'moral_action': item['moral_action'],'immoral_action': item['immoral_action']})
            judgement_history.append(f"{iter_count}th iteration,conflict-norm_judgement:{conflict_norm_judgement},situation_judgement:{situation_judgement} response_list:{response_list}")
            print(f"    situation_judgement:{situation_judgement}")
            if situation_judgement == 1 and conflict_norm_judgement == 1:
                break
            iter_count += 1
        else:
            # 修改conflict-norm
            instruction = instruction_4rec_confnorm_4oneclient
            prompt = get_prompt(instruction, {'situation': new_situation})
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_conflict_norm
            )
            print(f"    conflict-norm完成")
            Answer = completion.choices[0].message.content
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            conflict_norm_judgement = judge_4_conflict_norm(new_conflict_norm)
            print(f"    conflict-norm_judgement:{conflict_norm_judgement}")
            # 修改situation
            instruction = instruction_4rec_situation_4oneclient
            prompt = get_prompt(instruction, {'conflict-norm': new_conflict_norm})
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_situation
            )
            print(f"    situation完成")
            Answer = completion.choices[0].message.content
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
            response_list, situation_judgement = judge_4_situation({'situation': new_situation,'moral_action': item['moral_action'],'immoral_action': item['immoral_action']})
            print(f"    situation_judgement:{situation_judgement}")
            judgement_history.append(f"{iter_count}th iteration,conflict-norm_judgement:{conflict_norm_judgement},situation_judgement:{situation_judgement} response_list:{response_list}")
            if situation_judgement == 1 and conflict_norm_judgement == 1:
                break
            iter_count += 1
    # 将judgement_history先变成一个字符串，再加入messages_history
    judgement_history_str = '\n'.join(judgement_history)
    messages_history.append({'role': 'judge', 'content': judgement_history_str})
    new_data.append({'ID': item['ID'],'norm': item['norm'], 'conflict-norm': new_conflict_norm,'situation': new_situation, 'intention':item['intention'],'moral_action': item['moral_action'],'immoral_action': item['immoral_action'],'iter_count': iter_count-1,'situ_judgement': situation_judgement,'conflict_norm_judgement': conflict_norm_judgement})
    messages_history_all.append(messages_history)
    # 每生成一个数据，就保存一次
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    with open(output_path_4_messages_history, 'w') as f:
        json.dump(messages_history_all, f, indent=4, ensure_ascii=False)

