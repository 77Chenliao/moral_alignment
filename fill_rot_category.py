import  json


data_all_path = "../datasets/moral_stories/moral_stories_full_with_category.json"
input_path = r"D:\value_align\datasets\new_datasets\moral_stories_full_with_category_split2\moral_stories_full_with_category_split2_filtered_by_human.json"

with open(data_all_path, 'r',encoding='utf-8') as f:
    data_all = json.load(f)

with open(input_path, 'r',encoding='utf-8') as f:
    input_data = json.load(f)

rot_category = {}
for item in data_all:
    rot_category[item['ID']] = item['rot_category']


for item in input_data:
    item['rot_category'] = rot_category[item['ID']]

with open(input_path, 'w',encoding='utf-8') as f:
    json.dump(input_data, f, indent=4, ensure_ascii=False)