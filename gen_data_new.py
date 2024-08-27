import dashscope
from dashscope.api_entities.dashscope_response import Role
from utils import ali_api_key,openai_api_key,get_prompt, extract_moral_conflict_details
dashscope.api_key = ali_api_key
api_key = openai_api_key   # Fdu
import json
from openai import OpenAI
import os
from tqdm import tqdm
from instructions_new import instruction_4_background, instruction_4_shot, instruction_4_task_description


input_dir = "../datasets/moral_stories"
dataset_name = 'moral_stories_full_with_category_split1'
data_path = f"{input_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

model_name = 'gpt-4o-2024-08-06'

client = OpenAI(api_key=api_key)

strategy = 'basic'
output_dir = f"../datasets/new_datasets/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{dataset_name}_{strategy}.json"
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
else:
    new_data = []


Temperature_4_generation = 1.2

existing_data_len = len(new_data)
for item in tqdm(data[existing_data_len:]):
    input_token = 0
    output_token = 0
    messages_history = []
    instruction = '\n\n'.join([instruction_4_background, instruction_4_task_description, instruction_4_shot])
    prompt = get_prompt(instruction, item)
    messages_history.append({'role': Role.SYSTEM, 'content': prompt})
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages_history,
        temperature=Temperature_4_generation,
    )
    input_token += completion.usage.prompt_tokens
    output_token += completion.usage.completion_tokens
    Answer = completion.choices[0].message.content
    new_situation, conflict_norm, conflict_action = extract_moral_conflict_details(Answer)
    new_data.append({'ID': item['ID'], 'situation': item['situation'], 'new_situation': new_situation, 'norm': item['norm'],'action': item['moral_action'], 'conflict-norm': conflict_norm, 'conflict-action': conflict_action})
    # 即时保存
    with open(output_path, 'w',encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

