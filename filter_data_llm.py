import dashscope
from dashscope.api_entities.dashscope_response import Role
from utils import ali_api_key,openai_api_key,get_prompt, extract_filtering_result
dashscope.api_key = ali_api_key
api_key = openai_api_key   # Fdu
import json
from openai import OpenAI
import os
from tqdm import tqdm
from instructions import instruction_4_filtering
data_dir = "../datasets/new_datasets"
dataset_name = 'moral_stories_full_with_category_split1'
input_path = f"{data_dir}/{dataset_name}/{dataset_name}.json"
output_path = f"{data_dir}/{dataset_name}/{dataset_name}_filtered_by_llm.json"
with open(input_path, 'r',encoding='utf-8') as f:
    data = json.load(f)

model_name = 'gpt-4o-2024-08-06'

client = OpenAI(api_key=api_key)


if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        data_filtered = json.load(f)
    print(f"已经存在{len(data_filtered)}条数据,从上次断点继续")
else:
    data_filtered = []


Temperature = 1

system_description = "You're a strict data filtering model that filters out data that doesn't satisfy the conditions."

existing_data_len = len(data_filtered)
for item in tqdm(data[existing_data_len:]):
    messages_history = []
    prompt = get_prompt(instruction_4_filtering, item)
    messages_history.append({'role': Role.SYSTEM, 'content': system_description})
    messages_history.append({'role': Role.USER, 'content': prompt})
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages_history,
        temperature=Temperature,
    )
    Answer = completion.choices[0].message.content
    # input_token = completion.usage.prompt_tokens
    # output_token = completion.usage.completion_tokens
    # print(f"in:{input_token}, out:{output_token}")
    result, detail = extract_filtering_result(Answer)
    item['llm_evaluation'] = result
    item['llm_evaluation_detail'] = detail
    data_filtered.append(item)
    # 即时保存
    with open(output_path, 'w',encoding='utf-8') as f:
        json.dump(data_filtered, f, indent=4, ensure_ascii=False)

