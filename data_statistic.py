import json



data_path = r'D:\value_align\datasets\new_datasets\moral_stories_full_with_category\moral_stories_full_with_category_generated_iteratively_1client_gpt-4o.json'
import pandas as pd

with open(data_path, 'r') as f:
    data = json.load(f)


# 转换为DataFrame
df = pd.DataFrame(data)

# 筛选符合条件的项
filtered_df = df[(df['conflict_norm_judgement'] == 1) & (df['wrong_model_num'] != 0)]

# 分类并统计各级别数量
result = filtered_df.groupby(['rot_category', pd.cut(filtered_df['wrong_model_num'], bins=[0, 1, 3, 5], labels=['1', '2-3', '4-5'])]).size().unstack(fill_value=0)

# 统计符合条件的总数
total_count = len(filtered_df)

# 输出结果到文件
# result.to_csv('conflict_norm_judgement_analysis.csv')

# 打印结果
print("Total count of matching items:", total_count)
print(result)