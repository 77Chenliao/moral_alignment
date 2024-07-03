import json 
from utils import extract_ab

data_dir = r'/expe_on_moralstories/test_results/moral_dilemma_from_477'
model_name = 'gpt-3.5-turbo'

with_data_path = rf'{data_dir}\{model_name}_with_conflict.json'
without_data_path = rf'{data_dir}\{model_name}_without_conflict.json'

with open(with_data_path, 'r') as f:
    with_data = json.load(f)
with open(without_data_path, 'r') as f:
    without_data = json.load(f)

different_list = []

for i in range(len(with_data)):
    without_answer = extract_ab(without_data[i]['dilemma_answer'])
    without_label = without_data[i]['dilemma_label']
    if without_answer == without_label:
        original_true = True
    else:
        original_true = False
    with_answer = extract_ab(with_data[i]['dilemma_answer'])
    with_label = with_data[i]['dilemma_label']
    if with_answer != with_label:
        dilemma_true = False
    else:
        dilemma_true = True
    # 原本是对的，加了conflict-norm之后变错了
    if original_true and not dilemma_true:
        different_list.append(with_data[i]['ID'])

wrong_file = rf'{data_dir}\{model_name}_different_list.txt'
with open(wrong_file, 'w') as f:
    for item in different_list:
        f.write(str(item)+'\n')
