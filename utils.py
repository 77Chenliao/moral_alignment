import re

"""
def extract_norm(answer):
    answer = answer.replace("：", ":")
    answer = answer.replace(": ", ":")
    answer = re.sub(r'\s*:\s*', ':', answer)  # 更严格的处理冒号和空格
    answer = answer.replace("'s ", "’s ")
    answer = answer.replace(r"\"", "'")
    
    pattern_1 = r"(could be|may be|might be) '(.*?)'"
    pattern_2 = r"conflict-norm is:\s*'(.*?)'"
    pattern_3 = r"(could be|may be|might be)'(.*?)'"
    pattern_4 = r":'(.*?)'"  # 匹配引号中的内容
    pattern_5 = r"conflict-norm is:\s*(.*?)(?:,|\.|\?|!|$)"  # 匹配没有引号的情况
    pattern_6 = r"conflict-norm is: '(.*?)'"  # 处理可能的引号情况
    
    result_1 = re.search(pattern_1, answer)
    result_2 = re.search(pattern_2, answer)
    result_3 = re.search(pattern_3, answer)
    result_4 = re.search(pattern_4, answer)
    result_5 = re.search(pattern_5, answer)
    result_6 = re.search(pattern_6, answer)
    
    if result_1:
        result = result_1.group(2).strip()
    elif result_2:
        result = result_2.group(1).strip()
    elif result_3:
        result = result_3.group(2).strip()
    elif result_4:
        result = result_4.group(1).strip()
    elif result_5:
        result = result_5.group(1).strip()
    elif result_6:
        result = result_6.group(1).strip()
    else:
        return answer

    result = result.replace("’s ", "'s ")
    result = result.replace('\\"', '')
    result = result.replace('’', "'")
    if result[-1] in [',', '.', '?', '!']:
        return result[:-1] + '.'
    return result + '.'
"""
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



def extract_ab(answer):
    # 兼容之前的
    if answer =='A':
        return 'A'
    if answer == 'B':
        return 'B'
    # 利用re统计'(A)'的次数
    A_count = len(re.findall(r'\(A\)', answer))
    B_count = len(re.findall(r'\(B\)', answer))
    if A_count == 0 and B_count == 0:
        return None
    if A_count > B_count:
        return 'A'
    else:
        return 'B'
    
def extract_yesno_label(answer):
    # 全部小写，去除首尾空格
    answer = answer.lower().strip()
    if 'yes' in answer:
        return 1
    elif 'no' in answer:
        return 0
    else:
        return -1
    
def extract_moral_immoral_label(answer):
    # 全部小写，去除首尾空格
    answer = answer.lower().strip()
    if 'immoral' in answer:
        return 'immoral'
    else:
        return 'moral'
    

# main文件才运行
if __name__ == '__main__':
    answer = "Let's break it down step by step:\n\n### Step 1: Analyze the established norm and action\n\nThe established norm is: \"It's wrong to take credit for the work of others.\" This promotes honesty and integrity in acknowledging the origins of solutions and ideas.\n\n**Action:** Kevin uses the free solution and tells his boss where he found it from. This action aligns with the norm by being transparent about the source of the solution.\n\n### Step 2: Analyze the conflict-action\n\n**Conflict-action:** Kevin implements the free solution and tells his boss that it was all his idea. This goes against the norm by falsely claiming credit for someone else's work.\n\n### Step 3: Create a norm that leads to a moral conflict\n\nTo create a norm that leads to moral conflict, we need a principle that would justify Kevin's decision to claim the solution as his own. For instance, the norm could emphasize the importance of advancing within one's career to provide a better future for oneself and potentially one's family.\n\nA potential conflict-norm could be: \"In a competitive work environment, taking strategic credit for solutions can be essential for career advancement.\"\n\n### Step 4: Check the result\n\nKevin feels conflicted because while one norm directs him to be honest and transparent about the origins of solutions, the other encourages him to take strategic credit to advance his career. In adhering to one norm, he inevitably violates the other, leading to a complex moral situation.\n\nSo the conflict-norm is: \"In a competitive work environment, taking strategic credit for solutions can be essential for career advancement.\"\n\n### Final Output\n\n**Situation:** Kevin is working on a project at work when he sees that someone has already created a solution that is free online. Kevin wants to use the free solution.\n\n**Norm:** It's wrong to take credit for the work of others.\n\n**Action:** Kevin takes the solution and uses it, then tells his boss where he found it from.\n\n**Conflict-action:** Kevin implements the free solution and tells his boss that it was all his idea.\n\n**So the conflict-norm is:** In a competitive work environment, taking strategic credit for solutions can be essential for career advancement."
    print(extract_norm(answer))