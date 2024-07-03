import json
from utils import extract_norm

input_path =  r'D:\value_align\datasets\moral_stories\moral_stories_50\norm+context+immoral_action+newCOT_gpt-4o.json'

with open(input_path, 'r') as f:
    data = json.load(f)

for item in data:
    # print(item['ID'])
    answer = item['answer']
    item['conflict-norm'] = extract_norm(answer)

with open(input_path, 'w') as f:
    json.dump(data, f, indent=4)



import json

with open(input_path, 'r') as f:
    data = json.load(f)

wrong_item_list = []
for item in data:
    conflict_norm_word_count = len(item['conflict-norm'].split())
    if conflict_norm_word_count>30:
        wrong_item_list.append(item['ID'])

print(f"提取错误率: {len(wrong_item_list)/len(data)}")

wrong_file = r'/expe_on_moralstories/data_temp/problem_list.txt'
with open(wrong_file, 'w') as f:
    for item in wrong_item_list:
        f.write(str(item)+'\n')