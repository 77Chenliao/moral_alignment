import re


ali_api_key = "xx"
openai_api_key = 'xx'
dash_api_keys = ["xx", "xx","xx"]

def extract_norm(answer):
    if 'conflict-norm is:' in answer:
        result = answer.split('conflict-norm is:')[1].strip()
        result = result.split('.')[0].strip()
        result = result.replace('*', '')
        return result
    return answer

def extract_situation(answer):
    if 'Situation:' in answer:
        result = answer.split('Situation:')[1].strip()
        result = result.split('Norm:')[0].strip()
        return result
    return answer
        



def get_prompt(instruction, item):
    prompt = instruction
    if "{Norm}" in instruction:
        prompt = prompt.replace('{Norm}', item['norm'])
    if "{Situation}" in instruction:
        prompt = prompt.replace('{Situation}', item['situation'])
    if "{Intention}" in instruction:
        prompt = prompt.replace('{Intention}', item['intention'])
    if "{Moral_action}" in instruction:
        prompt = prompt.replace('{Moral_action}', item['moral_action'])
    if "{Moral_consequence}" in instruction:
        prompt = prompt.replace('{Moral_consequence}', item['moral_consequence'])
    if "{Immoral_action}" in instruction:
        prompt = prompt.replace('{Immoral_action}', item['immoral_action'])
    if "{Immoral_consequence}" in instruction:
        prompt = prompt.replace('{Immoral_consequence}', item['immoral_consequence'])
    if "{Conflict-norm}" in instruction:
        prompt = prompt.replace('{Conflict-norm}', item['conflict-norm'])
    return prompt




def extract_AB(answer):
    # 基本涵盖所有形式的输出
    if 'A.' in answer or 'A ' in answer or 'A,'in answer:
        return 'A'
    elif 'B.' in answer or 'B ' in answer or 'B,'in answer:
        return 'B'
    elif 'A' in answer:
        return 'A'
    elif 'B' in answer:
        return 'B'
    else:
        return None
    

# main文件才运行
if __name__ == '__main__':
    temp_answer = "Let's break it down step by step:\n\n### Step 1: Analyze the established norm and action\n\nThe established norm is: \"It's wrong to take credit for the work of others.\" This promotes honesty and integrity in acknowledging the origins of solutions and ideas.\n\n**Action:** Kevin uses the free solution and tells his boss where he found it from. This action aligns with the norm by being transparent about the source of the solution.\n\n### Step 2: Analyze the conflict-action\n\n**Conflict-action:** Kevin implements the free solution and tells his boss that it was all his idea. This goes against the norm by falsely claiming credit for someone else's work.\n\n### Step 3: Create a norm that leads to a moral conflict\n\nTo create a norm that leads to moral conflict, we need a principle that would justify Kevin's decision to claim the solution as his own. For instance, the norm could emphasize the importance of advancing within one's career to provide a better future for oneself and potentially one's family.\n\nA potential conflict-norm could be: \"In a competitive work environment, taking strategic credit for solutions can be essential for career advancement.\"\n\n### Step 4: Check the result\n\nKevin feels conflicted because while one norm directs him to be honest and transparent about the origins of solutions, the other encourages him to take strategic credit to advance his career. In adhering to one norm, he inevitably violates the other, leading to a complex moral situation.\n\nSo the conflict-norm is: \"In a competitive work environment, taking strategic credit for solutions can be essential for career advancement.\"\n\n### Final Output\n\n**Situation:** Kevin is working on a project at work when he sees that someone has already created a solution that is free online. Kevin wants to use the free solution.\n\n**Norm:** It's wrong to take credit for the work of others.\n\n**Action:** Kevin takes the solution and uses it, then tells his boss where he found it from.\n\n**Conflict-action:** Kevin implements the free solution and tells his boss that it was all his idea.\n\n**So the conflict-norm is:** In a competitive work environment, taking strategic credit for solutions can be essential for career advancement."
    print(extract_norm(temp_answer))