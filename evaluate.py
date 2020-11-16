#!/usr/bin/python
# coding:utf8

import json


def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(pre_lines, gold_lines):

        pre = pre["label"].get(label, {}).keys()
        gold = gold["label"].get(label, {}).keys()
        for i in pre:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pre:
                FN += 1

    p = TP / (TP + FP + 1e-20)
    r = TP / (TP + FN + 1e-20)
    f = 2 * p * r / (p + r + 1e-20)
    print('label: {}\nTP: {}\tFP: {}\tFN: {}'.format(label, TP, FP, FN))
    print('P: {:.2f}\tR: {:.2f}\tF1: {:.2f}'.format(p, r, f))
    print()
    return f


def get_f1_score(pre_file, gold_file):
    pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum += f
    avg = sum / (len(labels) + 1e-20)
    return f_score, avg


if __name__ == "__main__":
    # f_score, avg = get_f1_score(pre_file="test_predict.json", gold_file="test_gold.json")
    f_score, avg = get_f1_score(pre_file="cluener_bios/dev.json", gold_file="cluener_bios/dev.json")

    print(f_score, avg)