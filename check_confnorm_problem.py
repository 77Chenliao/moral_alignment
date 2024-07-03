import json
input_path = r'xx'

with open(input_path, 'r') as f:
    data = json.load(f)

# 提取问题
wrong_extract_list = []
wrong_gen_list = []
for item in data:
    conflict_norm_word_count = len(item['conflict-norm'].split())
    if 'sorry' in item['conflict-norm'] and conflict_norm_word_count > 30:
        wrong_gen_list.append(item['ID'])
    elif conflict_norm_word_count > 30 or conflict_norm_word_count < 6:
        wrong_extract_list.append(item['ID'])

print(f"提取错误率: {len(wrong_extract_list)/len(data)}")
print(f"生成错误率: {len(wrong_gen_list)/len(data)}")



wrong_file = r'/expe_on_moralstories/data_temp/problem_list.txt'
with open(wrong_file, 'w',encoding='utf-8') as f:
    if len(wrong_extract_list) !=0:
        f.write("提取错误：\n")
        for item in wrong_extract_list:
            f.write(str(item)+'\n')
    if len(wrong_gen_list) !=0:
        f.write("生成错误：\n")
        for item in wrong_gen_list:
            f.write(str(item)+'\n')