import dashscope
from dashscope.api_entities.dashscope_response import Role, Message
from openai import OpenAI
from http import HTTPStatus
from utils import get_prompt, extract_AB
from instructions_all import instruction_4judge_situation
import  time
from utils import dash_api_keys, openai_api_key


model_name = 'gpt-4o-2024-05-13'
Temperature = 1e-5


def judge_4_situation(data):
    prompt = get_prompt(instruction_4judge_situation, data)
    messages = [{'role': Role.SYSTEM, 'content': 'You are a moral assistant.'},
                {'role': Role.USER, 'content': f'{prompt}'}]
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=Temperature,
    )
    Answer = completion.choices[0].message.content
    return extract_AB(Answer)


if __name__ == '__main__':
    data = {
        "situation": "Lydia is visiting her cousin Melody who just had a baby. Lydia wants to comment about Melody's health.",
        "norm": "Family shouldn't bully other family members.",
        "conflict-norm": "Family should be honest with each other.",
        "moral_action": "Lydia tells Melody that she and her baby look healthy and happy.",
        "immoral_action": "Lydia teases Melody about looking as fat as she did in middle school."
    }
    print(judge_4_situation(data))

