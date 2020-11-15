import json
from collections import defaultdict


def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
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
    print(TP, FP, FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)
    print(p, r, f)
    return f


def get_f1_score(pre_lines, gold_file="data/thuctc_valid.json"):
    gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum += f
    avg = sum / len(labels)
    return f_score, avg


def convert_bios_to_json_lines(file_name):
    label_dict = defaultdict(lambda: defaultdict(list))
    json_lines = []
    flag = False
    entity = 0
    with open(file_name, 'r') as f:
        lines = f.readlines()
        words = []
        idx = 0
        for l in lines:
            if len(l.strip()) == 0:  # one sentence finished
                if flag:
                    label_dict[entity_name]["".join(words[start_idx: idx])].append([start_idx, idx - 1])
                    entity += 1
                    flag = False

                original_dict = {'text': "".join(words), "label": {k: dict(v) for k, v in label_dict.items()}}
                # print(original_dict)
                json_lines.append(original_dict)
                words = []
                idx = 0
                # print(label_dict)
                label_dict.clear()
                continue
            character, tag = l.strip().split(" ")
            # print(character, tag)
            words.append(character)
            if tag.startswith("B-") and not flag:
                entity_name = tag.split("-")[-1]  # get tag entity name
                start_idx = idx
                flag = True
            elif tag.startswith("B-") and flag:
                # print(entity_name, "".join(words[start_idx:idx]))
                label_dict[entity_name]["".join(words[start_idx: idx])].append([start_idx, idx - 1])
                entity += 1
                entity_name = tag.split("-")[-1]  # get tag entity name
                start_idx = idx
                flag = True
            elif tag.startswith("S-"):  # Single word
                entity_name = tag.split("-")[-1]  # get tag entity name
                label_dict[entity_name][character].append([idx, idx])
                entity += 1
                flag = False
            elif (tag.startswith("O") or tag.startswith("S-")) and flag:
                # print(entity_name, "".join(words[start_idx:idx]))
                label_dict[entity_name]["".join(words[start_idx: idx])].append([start_idx, idx - 1])
                entity += 1
                flag = False
            idx += 1
    return json_lines


if __name__ == '__main__':
    # test evaluation score
    print(get_f1_score(pre_lines=[json.loads(line.strip()) for line in open("cluener_bios/dev.json") if line.strip()],
          gold_file="cluener_bios/dev.json"))
    print(get_f1_score(pre_lines=convert_bios_to_json_lines("cluener_bios/dev.txt"), gold_file="cluener_bios/dev.json"))
