import json 

input_path =  r'D:\value_align\datasets\moral_stories\moral_stories_50\norm+context+immoral_action+newCOT_gpt-4o.json'
output_path = rf'{input_path[:-5]}_human_label.json'

with open(input_path, 'r') as f:
    data = json.load(f)

for item in data:
    # 清屏
    import os
    os.system('cls')
    item['human_label'] = None
    print("\n"+"situation:"+item['situation']+" "+item['intention']+"\n")
    print("norm:"+item['norm'])
    print("action:"+item['moral_action']+"\n")
    print("conflict-norm:"+item['conflict-norm'])
    print("conflict-action:"+item['immoral_action']+"\n")
    label = int(input("Please input the human label for this conflict-norm: 0 for unsatisfied, 1 for satisfied:  "))
    item['human_label'] = label

with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

# 统计1的数量，计算正确率
count = 0
for item in data:
    if item['human_label'] == 1:
        count += 1
correct_rate = count/len(data)
print(f"Correct rate: {correct_rate}")
print(f"Human label has been saved to {output_path}")
