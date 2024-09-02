# 导入json文件
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
import json
with open(r'C:\Users\84671\Desktop\code\datasets\new_datasets\moral_stories_full_with_category_split2\moral_stories_full_with_category_split2_filtered_by_human.json', 'r',encoding='utf-8') as f:
    data_human = json.load(f)

with open(r'C:\Users\84671\Desktop\code\datasets\new_datasets\moral_stories_full_with_category_split2\moral_stories_full_with_category_split2_filtered_by_llm.json', 'r',encoding='utf-8') as f:
    data_llm = json.load(f)

#提取data_human里的human_evaluation
human_evaluation = [item['human_evaluation'] for item in data_human]
llm_evaluation = [item['filtering_result'] for item in data_llm]

#计算准确率
print('Human evaluation:', sum(human_evaluation))
print('LLM evaluation:', sum(llm_evaluation))
accuracy = accuracy_score(human_evaluation, llm_evaluation)
print('Accuracy:', accuracy)
recall = recall_score(human_evaluation, llm_evaluation)
print('Recall:', recall)
kappa = cohen_kappa_score(human_evaluation, llm_evaluation)
print('Kappa:', kappa)

