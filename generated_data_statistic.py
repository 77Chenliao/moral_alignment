import json
import pandas as pd

# 加载数据
data_path = r'D:\value_align\datasets\new_datasets\final_datasets\moral_conflicts.json'

with open(data_path, 'r',encoding='utf-8') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 按 rot_category分组并统计数量
result = df.groupby('rot_category').size()

# 输出结果
print(result)
