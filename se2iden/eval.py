import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel, AdamW
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
from tqdm import tqdm
import pickle
import numpy as np

def calculate_metrics(ground_truth, prediction):
    def interval_length(interval):
        return max(0, interval[1] - interval[0] + 1)

    def intersection(interval1, interval2):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return (start, end) if start <= end else None

    total_intersection_length = 0
    total_completely_overlapped = 0

    for pred_interval in prediction:
        for gt_interval in ground_truth:
            inter = intersection(pred_interval, gt_interval)
            if inter:
                total_intersection_length += interval_length(inter)
                if pred_interval == gt_interval:
                    total_completely_overlapped += 1

    total_gt_length = sum(interval_length(interval) for interval in ground_truth)
    total_pred_length = sum(interval_length(interval) for interval in prediction)

    all_intervals = ground_truth + prediction
    all_intervals.sort()
    merged_intervals = []
    current_start, current_end = all_intervals[0]

    for start, end in all_intervals[1:]:
        if start > current_end:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)

    merged_intervals.append((current_start, current_end))
    total_union_length = sum(interval_length(interval) for interval in merged_intervals)

    iou = total_intersection_length / total_union_length if total_union_length > 0 else 0
    io_gt = total_intersection_length / total_gt_length if total_gt_length > 0 else 0
    io_p = total_intersection_length / total_pred_length if total_pred_length > 0 else 0
    complete_overlap_ratio = total_completely_overlapped / len(prediction) if len(prediction) > 0 else 0

    return {
        "IoU": iou,
        "IoGT": io_gt,
        "IoP": io_p,
        "CompleteOverlapRatio": complete_overlap_ratio
    }

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes1=2, num_classes2=2):
        super(MultiTaskModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier1 = nn.Linear(self.roberta.config.hidden_size, num_classes1)
        self.classifier2 = nn.Linear(self.roberta.config.hidden_size, num_classes2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation

        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        return logits1, logits2


def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1


# 在加载时使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel('microsoft/codebert-base').to(device)
model.load_state_dict(torch.load('multi_task_codebert_model.pth'))
tokenizer = RobertaTokenizer.from_pretrained('multi_task_codebert_model')

all_metrics = {'IoU': [], 'IoGT': [], 'IoP': [], 'CompleteOverlapRatio': []}
# Testing loop
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    with open("manual_dataset.jsonl", 'r') as fin, open("./res/manual_dataset.jsonl", 'w') as fout:
        for line in tqdm(fin, desc="Testing"):
            data = json.loads(line)
            raw_code = data["raw_code"]
            lines = raw_code.splitlines()
            all_preds, all_labels = [], []
            for code_snippet in data["code_snippets"]:
                all_labels.append(code_snippet["place"])
            cur_lines = []
            start, end = 0, 0
            for i in range(1, len(lines)):
                if lines[i].strip().startswith("//") or lines[i] == "\n" or lines[i].strip() == "":
                    continue
                if len(cur_lines) == 0:
                    cur_lines.append(lines[i])
                    start = i + 1
                    continue
                else:
                    text1 = "\n".join(cur_lines).strip()
                    text2 = lines[i].strip()
                    input_text = f"<CLS> {text1} <SEP> {text2} <EOS>"

                    with torch.no_grad():
                        encoding = tokenizer.encode_plus(
                            input_text,
                            add_special_tokens=True,
                            max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )

                        input_token = {
                            'input_ids': encoding['input_ids'],
                            'attention_mask': encoding['attention_mask']
                        }

                        input_ids = input_token['input_ids'].to(device)
                        attention_mask = input_token['attention_mask'].to(device)

                        logits1, logits2 = model(input_ids, attention_mask)
                        preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
                        preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
                        if int(preds1[0]) == 1:
                            if int(preds2[0]) == 1:
                                all_preds.append([start, end])
                            cur_lines = []
                            cur_lines.append(lines[i])
                            start = i + 1
                            end = i + 1
                        else:
                            end = i + 1
                            cur_lines.append(lines[i])
            metrics = calculate_metrics(all_labels, all_preds)
            all_metrics["IoU"].append(metrics["IoU"])
            all_metrics["IoGT"].append(metrics["IoGT"])
            all_metrics["IoP"].append(metrics["IoP"])
            all_metrics["CompleteOverlapRatio"].append(metrics["CompleteOverlapRatio"])
            data["res"] = {}
            data["res"]["all_preds"] = all_preds
            data["res"]["all_labels"] = all_labels
            data["res"]["metrics"] = metrics
            json.dump(data, fout)
            fout.write('\n')

with open("all_metrics.pkl", "wb") as file:
    pickle.dump(all_metrics, file)
print(np.mean(all_metrics["IoU"]))
print(np.mean(all_metrics["IoGT"]))
print(np.mean(all_metrics["IoP"]))
print(np.mean(all_metrics["CompleteOverlapRatio"]))
print("Finish")


# Example test:
# data = {
# 	"repo": "Wangqqingwen/openAGV",
# 	"path": "opentcs/openTCS-PlantOverview/src/main/java/org/opentcs/guing/components/dialogs/VehiclesPanel.java",
# 	"star_count": 21,
# 	"raw_code": "public void setVehicleModels(Collection<VehicleModel> vehicleModels) {\n    // Remove vehicles of the previous model from panel\n    for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.remove(vehicleView);\n    }\n\n    // Remove vehicles of the previous model from list\n    vehicleViews.clear();\n    // Add vehicles of actual model to list\n    for (VehicleModel vehicle : vehicleModels) {\n      vehicleViews.add(vehicleViewFactory.createSingleVehicleView(vehicle));\n    }\n\n    // Add vehicles of actual model to panel, sorted by name\n    for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.add(vehicleView);\n    }\n\n    panelVehicles.revalidate();\n  }",
# 	"code_summary": "/**\n   * Initializes this panel with the current vehicles.\n   *\n   * @param vehicleModels The vehicle models.\n   */",
# 	"code_snippets": [{
# 		"sub_id": 0,
# 		"code_snippet": "for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.remove(vehicleView);\n    }",
# 		"code_summary": "// Remove vehicles of the previous model from panel",
# 		"place": [3, 5]
# 	}, {
# 		"sub_id": 1,
# 		"code_snippet": "vehicleViews.clear();",
# 		"code_summary": "// Remove vehicles of the previous model from list",
# 		"place": [8, 8]
# 	}, {
# 		"sub_id": 2,
# 		"code_snippet": "for (VehicleModel vehicle : vehicleModels) {\n  vehicleViews.add(vehicleViewFactory.createSingleVehicleView(vehicle));\n}",
# 		"code_summary": "// Add vehicles of actual model to list",
# 		"place": [10, 12]
# 	}, {
# 		"sub_id": 3,
# 		"code_snippet": "for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.add(vehicleView);\n    }",
# 		"code_summary": "// Add vehicles of actual model to panel, sorted by name",
# 		"place": [15, 17]
# 	}],
# 	"id": 5044
# }
# raw_code = data["raw_code"]
# lines = raw_code.splitlines()
# all_preds, all_labels = [], []
# for code_snippet in data["code_snippets"]:
#     all_labels.append(code_snippet["place"])
# cur_lines = []
# start, end = 0, 0
# for i in range(1, len(lines)):
#     if lines[i].strip().startswith("//") or lines[i] == "\n" or lines[i].strip() == "":
#         continue
#     if len(cur_lines) == 0:
#         cur_lines.append(lines[i])
#         start = i+1
#         continue
#     else:
#         text1 = "\n".join(cur_lines).strip()
#         text2 = lines[i].strip()
#         input_text = f"<CLS> {text1} <SEP> {text2} <EOS>"
#
#         with torch.no_grad():
#             encoding = tokenizer.encode_plus(
#                 input_text,
#                 add_special_tokens=True,
#                 max_length=512,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#
#             input_token = {
#                 'input_ids': encoding['input_ids'],
#                 'attention_mask': encoding['attention_mask']
#             }
#
#             input_ids = input_token['input_ids'].to(device)
#             attention_mask = input_token['attention_mask'].to(device)
#
#             logits1, logits2 = model(input_ids, attention_mask)
#             preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
#             preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
#             print(preds1[0], preds2[0])
#             if int(preds1[0]) == 1:
#                 if int(preds2[0]) == 1:
#                     print(cur_lines)
#                     all_preds.append([start, end])
#                 cur_lines = []
#                 cur_lines.append(lines[i])
#                 start = i+1
#                 end = i+1
#             else:
#                 end = i+1
#                 cur_lines.append(lines[i])
# print(all_preds, all_labels)  # [[3, 5], [8, 8], [10, 12], [15, 17], [19, 19]] [[3, 5], [8, 8], [10, 12], [15, 17]]
# metrics = calculate_metrics(all_labels, all_preds)
# print(metrics)  # {'IoU': 0.9090909090909091, 'IoGT': 1.0, 'IoP': 0.9090909090909091, 'CompleteOverlapRatio': 0.8}