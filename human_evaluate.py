import json
import os

input_dir = "../datasets/new_datasets"
dataset_name = 'moral_stories_full_with_category_split1'
data_path = f"{input_dir}/{dataset_name}/{dataset_name}_generated_iteratively_1client_gpt-4o.json"
with open(data_path, 'r') as f:
    data = json.load(f)


output_path = data_path.split('.json')[0] + '_human_evaluated.json'
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        evaluated_data = json.load(f)
    print(f"已筛选{len(evaluated_data)}条数据")
else:
    evaluated_data = []

evaluated_len = len(evaluated_data)
for index, item in enumerate(data[evaluated_len:]):
    if item['level'] == 'easy' or item['conflict_norm_judgement'] == -1:
        item['human_evaluation'] = None
        evaluated_data.append(item)
    else:
        os.system('cls')
        print(f"{evaluated_len+index+1}/{len(data)}")
        print(f"Situation: {item['situation']}")
        print(f"Norm: {item['norm']}")
        print(f"Moral_action: {item['moral_action']}")
        print(f"Conflict-norm: {item['conflict-norm']}")
        print(f"Immoral_action: {item['immoral_action']}")
        label = input("OK? (1/x):")
        if label == '1':
            item['human_evaluation'] = 1
        else:
            item['human_evaluation'] = 0
        evaluated_data.append(item)
        with open(output_path, 'w') as f:
            json.dump(evaluated_data, f, indent=4)