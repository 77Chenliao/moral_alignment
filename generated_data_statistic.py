import json
import pandas as pd

# 加载数据
data_path = r'D:\value_align\datasets\new_datasets\moral_stories_full_with_category_split1\moral_stories_full_with_category_split1_generated_iteratively_1client_gpt-4o.json'

with open(data_path, 'r') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 筛选符合条件的项
filtered_df = df[df['conflict_norm_judgement'] == 1]

# 按 rot_category 和 level 分组并统计数量
result = filtered_df.groupby(['rot_category', 'level']).size().unstack(fill_value=0)

# 统计符合条件的总数
total_count = len(filtered_df)

result = result[['easy', 'medium', 'hard']]
# 打印结果
print(result)
print("Total count of matching items:", total_count)
proportions = result.sum() / result.sum().sum()
print("Proportions of matching items:", proportions)