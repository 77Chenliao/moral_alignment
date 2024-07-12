import requests
import json

# 读取数据
def judge_4_conflict_norm(conflict_norm):
    words = conflict_norm.split()
    action1 = '+'.join(words)
    url = f"https://mosaic-api-frontdoor.apps.allenai.org/predict?action1={action1}"
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            answer_text = response.json()
            c = answer_text['answer']['class']
            if c == 1:
                judgement = 1
            else:
                judgement = -1
            return judgement



if __name__ == '__main__':
    conflict_norm = "It is wrong to steal"
    c = judge_4_conflict_norm(conflict_norm)
    print(type(c))