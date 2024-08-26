import dashscope
from dashscope.api_entities.dashscope_response import Role
from utils import ali_api_key,openai_api_key,extract_norm, get_prompt, extract_situation
dashscope.api_key = ali_api_key
api_key = openai_api_key   # Fdu
import json
from openai import OpenAI
import os
from instructions_all import task_description,instruction_4gen_confnorm, instruction_4regen_confnorm, instruction_4finish_confnorm, instruction_4gen_situation, instruction_4regen_situation, instruction_4gen_another_confnorm, instruction_4gen_another_situation
from judge_4_situation import judge_4_situation
from judge_4_conflict_norm import judge_4_conflict_norm

input_dir = "../datasets/moral_stories"
dataset_name = 'moral_stories_full_with_category_split1'
data_path = f"{input_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

model_name = 'gpt-4o-mini-2024-07-18'

client = OpenAI(api_key=api_key)

strategy = 'basic'
output_dir = f"../datasets/new_datasets/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{dataset_name}_{strategy}.json"
output_path_4_messages_history = output_path.replace('.json', '_messages_history.json')
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
    with open(output_path_4_messages_history, 'r', encoding='utf-8') as f:
        messages_history_all = json.load(f)
else:
    new_data = []
    messages_history_all = []



ALL_ITER = 3 # 3次生成conflict-norm的机会
SITUATION_ITER = 3 # 每个conflict-norm下3次生成situation的机会



Temperature_4_conflict_norm = 1.2
Temperature_4_situation = 1.2


existing_data_len = len(new_data)
for index, item in enumerate(data[existing_data_len:]):
    messages_history = [] # 对话历史
    new_conflict_norm = ''
    new_situation = ''
    item['new_situation'] = 'not generated'
    item['conflict_norm'] = ''
    situation_judgement = -1
    conflict_norm_iteration_nums = 0
    qualified_conflict_norm_nums = 0
    situation_iteration_nums = 0
    original_length = len(f"{item['situation']}".split())
    DONE = False
    input_token_num = 0
    output_token_num = 0

    print(f"第{existing_data_len+index+1}/{len(data)}条数据开始生成")

    messages_history.append({'role': Role.SYSTEM, 'content': task_description})

    for iter_count in range(1, ALL_ITER+1):
        if DONE:
            break
        conflict_norm_iteration_nums += 1
        print(f"  第{iter_count }次迭代")
        if  iter_count == 1:
            instruction = instruction_4gen_confnorm
            prompt = get_prompt(instruction, item)
            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages_history,
                temperature=Temperature_4_conflict_norm
            )
            input_token_num += completion.usage.prompt_tokens
            output_token_num += completion.usage.completion_tokens
            Answer = completion.choices[0].message.content
            messages_history.append({'role': Role.ASSISTANT, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            item['conflict-norm'] = new_conflict_norm
            conflict_norm_judgement, conflict_norm_reason= judge_4_conflict_norm(item)
            if conflict_norm_judgement !=1: # 如果conflict-norm不合格，告知其理由
                print(f"    conflict-norm不合格")
                instruction_4regen_confnorm_with_reason = instruction_4regen_confnorm.replace('{Reason}',conflict_norm_reason)
                messages_history.append({'role': Role.USER, 'content': instruction_4regen_confnorm_with_reason})
                continue
            else:
                item['conflict-norm'] = new_conflict_norm
                print(f"    conflict-norm合格")
                qualified_conflict_norm_nums += 1
                messages_history.append({'role': Role.USER, 'content': instruction_4finish_confnorm})
                # 在该conflict-norm下迭代生成situation
                for i in range(1,SITUATION_ITER+1):
                    situation_iteration_nums += 1
                    if i == 1:
                        instruction = instruction_4gen_situation
                        prompt = get_prompt(instruction, item)
                        prompt = prompt.replace('{length_limit}', f"{max(100,2*original_length)}")
                        messages_history.append({'role': Role.USER, 'content': f'{prompt}'})
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=messages_history,
                            temperature=Temperature_4_situation
                        )
                        input_token_num += completion.usage.prompt_tokens
                        output_token_num += completion.usage.completion_tokens
                        Answer = completion.choices[0].message.content
                        messages_history.append({'role': Role.ASSISTANT, 'content': Answer})
                        new_situation = extract_situation(Answer)
                        item['new_situation'] = new_situation
                        situation_judgement, situation_reason= judge_4_situation(item)
                        if situation_judgement== 1:
                            DONE = True
                            print(f"      situation合格，stop")
                            break
                        else:
                            print(f"      situation不合格")
                    else:
                        instruction_4regen_situation_with_reason = instruction_4regen_situation.replace('{Reason}',situation_reason)
                        messages_history.append({'role': Role.USER, 'content': instruction_4regen_situation_with_reason})
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=messages_history,
                            temperature=Temperature_4_situation
                        )
                        input_token_num += completion.usage.prompt_tokens
                        output_token_num += completion.usage.completion_tokens
                        Answer = completion.choices[0].message.content
                        messages_history.append({'role': Role.ASSISTANT, 'content': Answer})
                        new_situation = extract_situation(Answer)
                        item['new_situation'] = new_situation
                        situation_judgement, situation_reason= judge_4_situation(item)
                        if situation_judgement == 1:
                            print(f"      situation合格，stop")
                            DONE = True
                            break
                        else:
                            print(f"      situation不合格")

                        if i == SITUATION_ITER: # 在conflict-norm下的所有situation都不合格，重新生成conflict-norm
                            instruction = instruction_4gen_another_confnorm
                            prompt = get_prompt(instruction, item)
                            messages_history.append({'role': Role.USER, 'content': f'{prompt}'})

        else:
            # 换一个conflict-norm继续迭代
            completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_history,
                    temperature=Temperature_4_conflict_norm
            )
            input_token_num += completion.usage.prompt_tokens
            output_token_num += completion.usage.completion_tokens
            Answer = completion.choices[0].message.content
            messages_history.append({'role': Role.ASSISTANT, 'content': Answer})
            new_conflict_norm = extract_norm(Answer)
            item['conflict-norm'] = new_conflict_norm
            conflict_norm_judgement, conflict_norm_reason = judge_4_conflict_norm(item)
            if conflict_norm_judgement !=1: # 如果conflict-norm不合格，重来
                print(f"    conflict-norm不合格")
                instruction_4regen_confnorm_with_reason = instruction_4regen_confnorm.replace('{Reason}',conflict_norm_reason)
                messages_history.append({'role': Role.USER, 'content': instruction_4regen_confnorm_with_reason})
                continue
            else:
                print(f"    conflict-norm合格")
                qualified_conflict_norm_nums += 1
                messages_history.append({'role': Role.USER, 'content': instruction_4finish_confnorm})
                # 在该conflict-norm下迭代生成situation
                prompt = get_prompt(instruction_4gen_another_situation, item)
                messages_history.append({'role': Role.USER, 'content': prompt.replace('{length_limit}', f"{max(100,2*original_length)}")})
                for i in range(1,SITUATION_ITER+1):
                    if i!=1:
                        instruction_4regen_situation_with_reason = instruction_4regen_situation.replace('{Reason}',situation_reason)
                        messages_history.append({'role': Role.USER, 'content': instruction_4regen_situation_with_reason})
                    situation_iteration_nums += 1
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages_history,
                        temperature=Temperature_4_situation
                    )
                    input_token_num += completion.usage.prompt_tokens
                    output_token_num += completion.usage.completion_tokens
                    Answer = completion.choices[0].message.content
                    messages_history.append({'role': Role.ASSISTANT, 'content': Answer})
                    new_situation = extract_situation(Answer)
                    item['new_situation'] = new_situation
                    situation_judgement, situation_reason = judge_4_situation(item)
                    if situation_judgement == 1:
                        print(f"      situation合格，stop")
                        DONE = True
                        break
                    else:
                        print(f"      situation不合格")
                    if i == SITUATION_ITER:
                        instruction = instruction_4gen_another_confnorm
                        prompt = get_prompt(instruction, item)
                        messages_history.append({'role': Role.USER, 'content': f'{prompt}'})




    new_data.append({'ID': item['ID'],'norm': item['norm'], 'conflict-norm': new_conflict_norm,'situation': item['situation'],"new_situation":item['new_situation'],'moral_action': item['moral_action'],'immoral_action': item['immoral_action'],'rot_category':item['rot_category'],'conflict_norm_judgement':conflict_norm_judgement, 'situation_judgement':situation_judgement, 'conflict_norm_iteration_nums': conflict_norm_iteration_nums, 'qualified_conflict_norm_nums': qualified_conflict_norm_nums, 'situation_iteration_nums': situation_iteration_nums})
    messages_history_all.append(messages_history)
    # 即时保存
    with open(output_path, 'w',encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    with open(output_path_4_messages_history, 'w', encoding='utf-8') as f:
        json.dump(messages_history_all, f, indent=4,ensure_ascii=False)

