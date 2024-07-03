import json 
import os 
from tqdm import tqdm

input_path = r'/datasets/moral_stories/moral_stories_full.json'
# llm_label_path = r'D:\value_align\datasets\moral_stories\moral_stories_50\moral_stories_50_immoral_action_labelwith_delphi.json'
# with open(llm_label_path, 'r') as f:
#     llm_data = json.load(f)
output_path = rf'{input_path[:-5]}_immoral_action_human_label.json'

with open(input_path, 'r',encoding='utf-8') as f:
    data = json.load(f)

if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        existing_data = json.load(f)
    print(f"已经存在{len(existing_data)}条数据,从上次断点继续")
else:
    existing_data = []

existing_data_len = len(existing_data)
for item in tqdm(data[existing_data_len:]):
    # 清屏
    import os
    print('\n')
    print("situation:"+item['situation'])
    print("intention:"+item['intention'])
    print("norm:"+item['norm'])
    print("moral-action:"+item['moral_action'])
    print("immoral-action:"+item['immoral_action'])
    label = input("Please input the human label for this immoral action: 1 for absolute immoral, 0 for else: ")
    if label =='1':
        label = 1
    else:
        label = 0
    item['immoral_action_human_label'] = label
    existing_data.append(item)
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    os.system('cls')

# 统计1的数量，计算正确率
# count = 0
# for item in data:
#     if item['immoral_action_human_label'] == 1:
#         count += 1
# correct_rate = count/len(data)
# print(f"filtered rate: {correct_rate}")
# print(f"Human label has been saved to {output_path}")
