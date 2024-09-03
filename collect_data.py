import json
import os

input_path = r'D:\value_align\datasets\new_datasets\moral_stories_full_with_category_split1\moral_stories_full_with_category_split1_filtered_by_human.json'

output_path = r"D:\value_align\datasets\new_datasets\final_datasets\moral_conflicts.json"

if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"已存在{len(data)}条数据")
else:
    data = []

existing_data_ids = set([item['ID'] for item in data])

with open(input_path, 'r',encoding='utf-8') as f:
    new_data = json.load(f)

new_num = 0
for item in new_data:
    if item['ID'] not in existing_data_ids and item['human_evaluation'] == 1:
        new_item = {"ID": item['ID'], "moral_conflict": item['moral_conflict'], "norm": item['norm'], "action": item['action'], "conflict-norm": item['conflict-norm'], "conflict-action": item['conflict-action'], "rot_category": item['rot_category']}
        data.append(new_item)
        new_num += 1
    else:
        pass

with open(output_path, 'w',encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"新增{new_num}条数据")

