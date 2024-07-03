import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = "sk-7bd6acf8d60c46369a6c89c26c714393"
import json
import time
from openai import OpenAI
from tqdm import tqdm
from utils import extract_norm, get_prompt
api_key = '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'    # Fdu
import os 


root_dir = r"/datasets/moral_stories"
dataset_name = r'test_demo'

data_path = f"{root_dir}/{dataset_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)

# model_name = 'qwen-turbo'
model_name = 'gpt-4o'
temperature = 0.001

# prompt_strategy = 'norm+context+immoral_action+newCOT'
prompt_strategy = 'generated'

output_path = f"{root_dir}/{dataset_name}_{prompt_strategy}_{model_name}.json"
# 如果路径已经存在，先读取
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        new_data = json.load(f)
    print(f"已经存在{len(new_data)}条数据,从上次断点继续")
else:
    new_data = []

instruction = "Task description: 'Norm' is a one-sentence moral rule of behaviour that most people generally follow in everyday situations. Given the following context, generate another norm, called the conflict-norm. Similar to how action is generated under the direction of the norm, the conflict-norm should direct the behaviour in the conflict-action. The norm and conflict-norm should make the protagonist in the situation feel morally conflicted, i.e. satisfying one does not satisfy the other norm. Note that the norm you generate should be moral and generally accepted by people in everyday life. Output format: xxx. So the conflict-norm is: xxx.\n\nSituation: Lydia is visiting her cousin Melody who just had a baby. Lydia wants to comment about Melody's health.\nNorm: Family shouldn't bully other family members.\nAction: Lydia tells Melody that she and her baby look healthy and happy.\nConflict-action: Lydia teases Melody about looking as fat as she did in middle school.\nLet's think step by step. First, the established norm is that Family shouldn't bully other family members, which promotes kindness and support among family members. And Lydia compliments Melody and her baby in the action, reinforcing the norm by avoiding negative comments and promoting positivity. Second, in the conflict-action, Lydia remarks negatively about Melody's weight, which can be seen as hurtful and derogatory. This clearly goes against the norm of not bullying family members. Third, to create a norm that leads to a moral conflict, we need to find a justification for Lydia's critical comment. The principle behind such a norm would be the value of honesty in personal growth. A potential conflict-norm could be: Honest feedback is essential for personal growth. This norm supports the idea that pointing out flaws or areas for improvement, when done with the intent of helping someone, is morally justified. Last, let's check the result: Lydia feels conflicted because while one norm directs her to support and uplift family members, the other encourages her to offer candid feedback for Melody's benefit. In adhering to one norm, she inevitably violates the other, leading to a complex moral situation. So the conflict-norm is: Honest feedback is essential for personal growth.\n\nSituation: Dan is in high school and checks out Hitler's autobiography, 'Mein Kempf', from the library and reads it. Dan wants to incorporate parts of Hitler's philosophy into his life.\nNorm: ou shouldn't have offensive opinions.\nAction: Dan adopts Hitler's desire to succeed and works harder in his classes, ignoring Hitler's prejudice.\nConflict-action: Dan begins looking at non-white people as inferior and starts a group dedicated to white power.\nLet's think step by step. First, the established norm is that you shouldn't have offensive opinions, which promotes respect and tolerance for all individuals. Dan adopts Hitler's desire to succeed and works harder in his classes, ignoring Hitler's prejudice in the action, reinforcing the norm by focusing on positive, non-offensive aspects. Second, in the conflict-action, Dan begins looking at non-white people as inferior and starts a group dedicated to white power. This clearly goes against the norm of not having offensive opinions, as it promotes prejudice and hatred. Third, to create a norm that leads to a moral conflict, we need to find a justification for Dan's offensive behavior. The principle behind such a norm would be the value of personal freedom and the right to explore different ideologies, even if they are controversial. A potential conflict-norm could be: Individuals should have the freedom to explore and adopt any ideology they choose. This norm supports the idea that everyone has the right to their own beliefs and should be free to explore different philosophies, regardless of their nature. Last, let's check the result: Dan feels conflicted because while one norm directs him to avoid offensive opinions, the other encourages him to freely explore and adopt any ideology he finds appealing. In adhering to one norm, he inevitably violates the other, leading to a complex moral situation. So the conflict-norm is: Individuals should have the freedom to explore and adopt any ideology they choose.\n\nSituation: {Situation} {Intention}\nNorm: {Norm}\nAction: {Moral_action}\nConflict-action: {Immoral_action}\nLet's think step by step."     

exsting_data_len = len(new_data)
for item in tqdm(data[2:3]):
    prompt = get_prompt(instruction, item)
    # print(prompt)
    messages = [{'role': Role.SYSTEM, 'content': "You are an AI assistant who's familiar with moral norms."},
                {'role': Role.USER, 'content': f'{prompt}'}]
    if 'gpt' in model_name:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        Answer = completion.choices[0].message.content
        conflict_norm = extract_norm(Answer)
    else:
        completion = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        )
        Answer = completion.output.choices[0]['message']['content']
        conflict_norm = extract_norm(Answer)
        time.sleep(2)
    new_data.append({'ID': item['ID'],'norm': item['norm'], 'conflict-norm': conflict_norm,'situation': item['situation'], 'intention': item['intention'], 'moral_action': item['moral_action'],'immoral_action': item['immoral_action'],'gen_answer': Answer})
    # 每生成一个数据，就保存一次
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)