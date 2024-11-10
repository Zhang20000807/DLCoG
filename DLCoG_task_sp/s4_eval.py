from tqdm import tqdm
from cbleu import *
from rouge import Rouge
import json
from nltk.translate.meteor_score import meteor_score
import re

from eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu,get_meteor,get_rouge,get_cider,get_nltk33_sent_bleu,get_google_sent_bleu


def word_tknz(s):
    pattern = r'[a-zA-Z]+'
    tokens = re.findall(pattern, s)
    item_str = [token.lower() for token in tokens]
    if len(item_str) == 0:
        item_str = ["a"]
    return item_str

def calculate_meteor_score(reference, hypothesis):
    score = meteor_score([reference], hypothesis)
    return score

with open("", 'r') as f:
    data = [json.loads(line) for line in f]
    snippet_gt, snippet_gpt, method_gt, method_gpt = [], [], [], []

    idx = 0
    scores, scores_o = [], []
    for item in tqdm(data, total=len(data)):
        raw_item = item
        item = item["res"]
        if item["result"]["method"]["gpt_res"] != "ERROR":
            method_gt.append(item["result"]["method"]["ground_truth"])
            method_gpt.append(item["result"]["method"]["gpt_res"])

            for snippet_data in item["result"]["snippet"]:
                if snippet_data["gpt_res"] != "ERROR":
                    snippet_gt.append(snippet_data["ground_truth"])
                    snippet_gpt.append(snippet_data["gpt_res"])

    print("Parser data number finished: ", len(method_gt), len(method_gpt))
    snippet_gt = [word_tknz(sent) for sent in snippet_gt]
    snippet_gpt = [word_tknz(sent) for sent in snippet_gpt]
    method_gt = [word_tknz(sent) for sent in method_gt]
    method_gpt = [word_tknz(sent) for sent in method_gpt]

    print("*" * 40, "snippet comment", "*" * 40)
    print("ROUGE: ", get_rouge(snippet_gt, snippet_gpt))
    print("Sn-BLEU: ", get_nltk33_sent_bleu(snippet_gt, snippet_gpt))
    print("Sg-BLEU: ", get_google_sent_bleu(snippet_gt, snippet_gpt))
    sum_meteor = 0
    cnt = 0
    for gt, gpt in zip(snippet_gt, snippet_gpt):
        cnt += 1
        sum_meteor += (calculate_meteor_score(gt, gpt) * 100)
    print("METEOR: ", sum_meteor / cnt)

    print("*"*40, "method comment", "*"*40)
    print("ROUGE: ", get_rouge(method_gt, method_gpt))
    print("Sn-BLEU: ", get_nltk33_sent_bleu(method_gt, method_gpt))
    print("Sg-BLEU: ", get_google_sent_bleu(method_gt, method_gpt))
    sum_meteor = 0
    cnt = 0
    for gt, gpt in zip(method_gt, method_gpt):
        cnt += 1
        sum_meteor += (calculate_meteor_score(gt, gpt) * 100)
    print("METEOR: ", sum_meteor / cnt)
