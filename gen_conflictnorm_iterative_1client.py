import dashscope
from dashscope.api_entities.dashscope_response import Role
from utils import ali_api_key,openai_api_key,extract_norm, get_prompt, extract_situation
dashscope.api_key = ali_api_key
api_key = openai_api_key   # Fdu
import json
from openai import OpenAI
import os
from instructions_all import instruction_4gen_confnorm, instruction_4rec_confnorm, instruction_4gen_situation, instruction_4rec_situation 
from judge_4_situation import judge_4_situation
from judge_4_conflict_norm import judge_4_conflict_norm

input_dir = "../datasets/moral_stories"
dataset_name = 'moral_stories_full_with_category'
data_path = f"{input_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

model_name = 'gpt-4o-2024-05-13'
client = OpenAI(api_key=api_key)

prompt_strategy = 'generated_iteratively_1client'
output_dir = f"../datasets/new_datasets/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{dataset_name}_{prompt_strategy}_gpt-4o.json"
output_path_4_messages_history = f"{output_dir}/{dataset_name}_{prompt_strategy}_gpt-4o_messages_history_all.json"
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
    with open(output_path_4_messages_history, 'r') as f:
        messages_history_all = json.load(f)
else:
    new_data = []
    messages_history_all = []



MAX_ITER = 3
Temperature_4_conflict_norm = 1
Temperature_4_situation = 1


existing_data_len = len(new_data)
for index, item in enumerate(data[existing_data_len:]):
    messages_history = []
    judgement_history = []
    new_conflict_norm = ''
    new_situation = ''
    min_diff = 10
    best_situation = ''
    best_conflict_norm = ''
    best_conflict_norm_judgement = -1 

    print(f"第{existing_data_len+index+1}/{len(data)}条数据开始生成")
    for iter_count in range(1, MAX_ITER+1):
        print(f"  第{iter_count }次迭代")
        if  iter_count == 1:
            # 首先生成conflict-norm)
            instruction = instruction_4gen_confnorm
            prompt = get_prompt(instruction, item)
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_conflict_norm
            )
            Answer = completion.choices[0].message.content
            print(f"    conflict-norm完成.")
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            item['conflict-norm'] = new_conflict_norm
            conflict_norm_judgement = judge_4_conflict_norm(new_conflict_norm)
            print(f"    conflict-norm_judgement:{conflict_norm_judgement}")
            # 然后扩充situation
            instruction = instruction_4gen_situation
            original_length = len(f"{item['situation']} {item['intention']}".split())
            prompt = get_prompt(instruction, item)
            prompt = prompt.replace('{length_limit}', f"{max(100,2*original_length)}")
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_situation
            )
            Answer = completion.choices[0].message.content
            print(f"    situation完成.")
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
            response_list, diff = judge_4_situation({'situation': new_situation,'norm': item['norm'],'conflict-norm':new_conflict_norm,'moral_action': item['moral_action'],'immoral_action': item['immoral_action']})
            judgement_history.append(f"{iter_count}th iteration,conflict-norm_judgement:{conflict_norm_judgement},response_list:{response_list}")
            print(f"    differ :{diff}")
            if diff < min_diff:
                min_diff = diff
                best_situation = new_situation
                best_conflict_norm = new_conflict_norm
                best_conflict_norm_judgement = conflict_norm_judgement
        else:
            # 修改conflict-norm
            instruction = instruction_4rec_confnorm
            prompt = get_prompt(instruction, {'situation': new_situation})
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_conflict_norm
            )
            Answer = completion.choices[0].message.content
            print(f"    conflict-norm完成.")
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            conflict_norm_judgement = judge_4_conflict_norm(new_conflict_norm)
            print(f"    conflict-norm_judgement:{conflict_norm_judgement}")
            # 修改situation
            instruction = instruction_4rec_situation
            prompt = get_prompt(instruction, {'conflict-norm': new_conflict_norm})
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_situation
            )
            Answer = completion.choices[0].message.content
            print(f"    situation完成.")
            messages_history.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
            response_list, diff = judge_4_situation({'situation': new_situation,'norm': item['norm'],'conflict-norm':new_conflict_norm,'moral_action': item['moral_action'],'immoral_action': item['immoral_action']})
            judgement_history.append(f"{iter_count}th iteration,conflict-norm_judgement:{conflict_norm_judgement},response_list:{response_list}")
            print(f"    differ :{diff}")
            if diff < min_diff:
                min_diff = diff
                best_situation = new_situation
                best_conflict_norm = new_conflict_norm
                best_conflict_norm_judgement = conflict_norm_judgement
    # 将judgement_history先变成一个字符串，再加入messages_history
    judgement_history_str = '\n'.join(judgement_history)
    messages_history.append({'role': 'judge', 'content': judgement_history_str})
    if min_diff == 5:
        level = 'easy'
    elif min_diff == 3:
        level = 'medium'
    else:
        level = 'hard'
    new_data.append({'ID': item['ID'],'norm': item['norm'], 'conflict-norm': best_conflict_norm,'situation': best_situation,'moral_action': item['moral_action'],'immoral_action': item['immoral_action'],'rot_category':item['rot_category'],'level':level,'conflict_norm_judgement':best_conflict_norm_judgement})
    messages_history_all.append(messages_history)
    # 即时保存
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    with open(output_path_4_messages_history, 'w') as f:
        json.dump(messages_history_all, f, indent=4)

