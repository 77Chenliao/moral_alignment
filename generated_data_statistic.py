import json
import pandas as pd

# 加载数据
data_path = r'D:\value_align\datasets\new_datasets\moral_stories_full_with_category_split1\moral_stories_full_with_category_split1_filtered_by_human.json'

with open(data_path, 'r') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 按 rot_category 和 level 分组并统计数量
result = df.groupby(['rot_category', 'level']).size().unstack(fill_value=0)

# 统计符合条件的总数
total_count = len(filtered_df)

result = result[['easy', 'medium', 'hard']]
# 打印结果
print(result)
print("Total count of matching items:", total_count)
proportions = result.sum() / result.sum().sum()
print("Proportions of matching items:", proportions)