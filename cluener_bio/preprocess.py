import json


def read_json(input_file):
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities is not None:
                all_spans = []
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            all_spans.append((key, sub_name, start_index, end_index))

                all_spans = sorted(all_spans, key=lambda x: x[3] - x[2])
                # print(all_spans)
                for key, sub_name, start_index, end_index in all_spans:
                    assert ''.join(words[start_index:end_index + 1]) == sub_name

                    # if start_index == end_index:
                    #     labels[start_index] = 'S-' + key
                    # else:
                    labels[start_index] = 'B-' + key
                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                    # if sub_name == '《招商银行：投资永隆银行浮亏逾百亿港元》':
                    # print(labels)
                    # assert 1 == 0
                lines.append({"words": words, "labels": labels})
            #         for sub_name, sub_index in value.items():
            #             for start_index, end_index in sub_index:
            #                 assert ''.join(words[start_index:end_index + 1]) == sub_name
            #                 if start_index == end_index:
            #                     labels[start_index] = 'S-' + key
            #                 else:
            #                     labels[start_index] = 'B-' + key
            #                     labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            # lines.append({"words": words, "labels": labels})
    return lines


files = [k + '.json' for k in ['dev', 'test', 'train']]
# files = ['train.json']
for f in files:
    lines = read_json(f)
    for l in lines:
        w, l = l['words'], l['labels']
        with open(f.replace(".json", ".txt"), "w") as fw:
            for tok, lbl in zip(w, l):
                fw.write(tok + " " + lbl + "\n")
            fw.write("\n")
