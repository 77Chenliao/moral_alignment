import json
import os

data_dir = "../datasets/new_datasets"
dataset_name = 'moral_stories_full_with_category_split1'
input_path = f"{data_dir}/{dataset_name}/{dataset_name}_filtered_by_llm.json"
output_path = f"{data_dir}/{dataset_name}/{dataset_name}_filtered_by_human.json"
with open(input_path, 'r',encoding='utf-8') as f:
    data = json.load(f)

if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        data_filtered = json.load(f)
    print(f"已经存在{len(data_filtered)}条数据,从上次断点继续")
else:
    data_filtered = []


existing_data_len = len(data_filtered)
for item in data[existing_data_len:]:
    if item['llm_evaluation'] == 0:
        item['human_evaluation'] = 0
        existing_data_len += 1
        data_filtered.append(item)
    else:
        print(f"{existing_data_len + 1}/{len(data)}")
        print(f"Moral conflict: {item['moral_conflict']}")
        print(f"Norm_1: {item['norm']}")
        print(f"Norm_2: {item['conflict-norm']}")
        print(f"Action_1: {item['action']}")
        print(f"Action_2: {item['conflict-action']}")
        # 即时保存
        result = input("Satisfied?(1/x):  ")
        if result == '1':
            item['human_evaluation'] = 1
        else:
            item['human_evaluation'] = 0
        data_filtered.append(item)
        existing_data_len += 1
    with open(output_path, 'w',encoding='utf-8') as f:
        json.dump(data_filtered, f, indent=4, ensure_ascii=False)

