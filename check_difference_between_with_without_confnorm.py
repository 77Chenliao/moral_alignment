import json 
import os 

data_dir = r'/expe_on_moralstories/test_results/moral_dilemma_from_477'
model_name = 'gpt-3.5-turbo'

with_data_path = rf'{data_dir}\{model_name}_with_conflict.json'
without_data_path = rf'{data_dir}\{model_name}_without_conflict.json'

with_data_id_map = {}
with open(with_data_path, 'r') as f:
    with_data = json.load(f)
for item in with_data:
    with_data_id_map[item['ID']] = item
without_data_id_map = {}
with open(without_data_path, 'r') as f:
    without_data = json.load(f)
for item in without_data:
    without_data_id_map[item['ID']] = item

wrong_file = rf'{data_dir}\{model_name}_different_list.txt'
with open(wrong_file, 'r') as f:
    wrong_list = f.readlines()
wrong_list = [item.strip() for item in wrong_list]

for id in wrong_list:
    without_item = without_data_id_map[id]
    with_item = with_data_id_map[id]
    os.system('cls')
    print(f'道德观念1:{without_item["norm"]}')
    print(f'道德观念2:{without_item["conflict-norm"]}')
    print(f'情境:{without_item["situation"]} {without_item["intention"]}')
    print(f'道德行为:{without_item["moral_action"]}')
    print(f'不道德行为:{without_item["immoral_action"]}')
    user_input = input('continue?')
        