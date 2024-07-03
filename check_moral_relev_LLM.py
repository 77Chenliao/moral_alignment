# 用于对大模型评估的结果进行检查
import json
from sklearn.metrics import precision_score, recall_score

with open(r'D:\value_align\datasets\moral_stories\moral_stories_50\norm+context+immoral_action+newCOT_gpt-3.5-turbo_human_label.json', 'r') as f:
    human_data = json.load(f)
labels = [item['human_label'] for item in human_data]

check_file = r'D:\value_align\datasets\moral_stories\moral_stories_50\norm+context+immoral_action+newCOT_gpt-3.5-turbo_labelwith_gpt-4-turbo.json'
with open(check_file, 'r') as f:
    check_data = json.load(f)

moral_predictions = [item['moral_label'] for item in check_data]
relev_predictions = [item['relev_label'] for item in check_data]

moral_precision = precision_score(labels, moral_predictions)
moral_recall = recall_score(labels, moral_predictions)
relev_precision = precision_score(labels, relev_predictions)
relev_recall = recall_score(labels, relev_predictions)

print(f"Moral precision: {moral_precision}, recall: {moral_recall}")
print(f"Relevance precision: {relev_precision}, recall: {relev_recall}")