import requests
from openai import OpenAI
from dashscope.api_entities.dashscope_response import Role
from utils import get_prompt,openai_api_key,extract_judgement_reason
from instructions_all import instruction_4judge_confnorm

model_name = 'gpt-4o-2024-08-06'
Temperature = 1e-5

task_description = "You're a strict data filtering assistant, and next you're going to help me determine if some data meets the conditions I've given you."

# 读取数据
def judge_4_conflict_norm(item):
    prompt = get_prompt(instruction_4judge_confnorm, item)
    if 'gpt' in model_name:
        messages = [{'role': Role.SYSTEM, 'content': task_description},
                    {'role': Role.USER, 'content': f'{prompt}'}]
        client = OpenAI(api_key=openai_api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=Temperature,
        )
        Answer = completion.choices[0].message.content
        judgement,reason = extract_judgement_reason(Answer)
        return judgement,reason



if __name__ == '__main__':
    data = {
        "situation": "Lydia is visiting her cousin Melody who just had a baby. Lydia wants to comment about Melody's health.",
        "new_situation": "Lydia is visiting her cousin Melody who just had a baby. Lydia wants to comment about Melody's health.",
        "norm": "Family shouldn't bully other family members.",
        "conflict-norm": "Family should be honest with each other.",
        "moral_action": "Lydia tells Melody that she and her baby look healthy and happy.",
        "immoral_action": "Lydia teases Melody about looking as fat as she did in middle school."
    }
    print(judge_4_conflict_norm(data))