import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = "sk-7bd6acf8d60c46369a6c89c26c714393"
import json
import time
from openai import OpenAI
from tqdm import tqdm
from utils import extract_norm, get_prompt, extract_situation
api_key = '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'    # Fdu
import os 
from gen_conflictnorm_iterative_instructions import instruction_4gen_confnorm, instruction_4rec_confnorm, instruction_4rec_situation, instruction_4gen_situation

root_dir = r"/datasets/moral_stories"
dataset_name = r'test_demo'

data_path = f"{root_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

model_name = 'gpt-4o'
prompt_strategy = 'generated_iteratively'

output_path = f"{root_dir}/{dataset_name}_{prompt_strategy}_{model_name}.json"
# 如果路径已经存在，先读取
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
else:
    new_data = []

messages_history_4_confnorm_all = []
messages_history_4_situation_all = []
MAX_ITER = 3
exsting_data_len = len(new_data)
for item in tqdm(data[2:3]):
    messages_history_4_confnorm = [{'role': Role.SYSTEM, 'content': "You are an AI assistant who's familiar with moral norms."}]
    messages_history_4_situation = [{'role': Role.SYSTEM, 'content': "You are an AI assistant who's familiar with moral norms."}]
    new_conflict_norm = ''
    new_situation = ''
    # 对每条数据迭代式生成
    for i in range(MAX_ITER):
        # 如果是第一次生成
        if i == 0:
            # 首先生成conflict-norm
            instruction = instruction_4gen_confnorm
            prompt = get_prompt(instruction, item)
            messages_history_4_confnorm.append({'role': Role.USER, 'content': f'{prompt}'})
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history_4_confnorm,
                    temperature=0.001
            )
            Answer = completion.choices[0].message.content
            # 将生成的conflict-norm加入历史
            messages_history_4_confnorm.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            item['conflict-norm'] = new_conflict_norm

            # 然后扩充situation
            instruction = instruction_4gen_situation
            original_length = len(f"{item['situation']} {item['intention']}".split())
            prompt = get_prompt(instruction, item)
            prompt = prompt.replace('{length_limit}', f"{min(70,1.5*original_length)}")
            messages_history_4_situation.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history_4_situation,
                    temperature=0.001
            )
            Answer = completion.choices[0].message.content
            # 将生成的situation加入历史
            messages_history_4_situation.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
        else:
            # 迭代修改
            instruction = instruction_4rec_confnorm
            prompt = get_prompt(instruction, {'situation': new_situation})
            messages_history_4_confnorm.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history_4_confnorm,
                    temperature=1
            )
            Answer = completion.choices[0].message.content
            # 将生成的conflict-norm加入历史
            messages_history_4_confnorm.append({'role': Role.SYSTEM, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            # 然后扩充situation
            instruction = instruction_4rec_situation
            prompt = get_prompt(instruction, {'conflict-norm': new_conflict_norm})
            messages_history_4_situation.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history_4_situation,
                    temperature=1
            )
            Answer = completion.choices[0].message.content
            # 将生成的situation加入历史
            messages_history_4_situation.append({'role': Role.SYSTEM, 'content': Answer})
            new_situation = extract_situation(Answer)
    new_data.append({'ID': item['ID'],'norm': item['norm'], 'conflict-norm': new_conflict_norm,'situation': new_situation, 'intention':item['intention'],'moral_action': item['moral_action'],'immoral_action': item['immoral_action']})
    # 每生成一个数据，就保存一次
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    messages_history_4_confnorm_all.append(messages_history_4_confnorm)
    messages_history_4_situation_all.append(messages_history_4_situation)

# 保存最终的对话历史
with open(f"{root_dir}/{dataset_name}_{prompt_strategy}_{model_name}_messages_history_4_confnorm.json", 'w') as f:
    json.dump(messages_history_4_confnorm_all, f, indent=4)
with open(f"{root_dir}/{dataset_name}_{prompt_strategy}_{model_name}_messages_history_4_situation.json", 'w') as f:
    json.dump(messages_history_4_situation_all, f, indent=4)