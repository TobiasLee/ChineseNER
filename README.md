# ChineseNER



## Prepare environment
```bash
    conda create -n ner python=3.6
    conda activate ner
    conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
    pip install -r requirements.txt 
```
## Download pre-trained Chinese BERT

> sh download.sh 

## Train & eval with BIOS metric
> sh ner_train.sh 

## Eval with official metric

> sh ner_eval.sh

## Result

**Result on the dev set (BIOS)**

|                              | loss           | accuracy       | precision       | recall          | f1               |
| ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- |
| BERT+Softmax+CE_loss         | 0.20 | 94.34    | 73.44     | 80.73     | 76.91 |
| BERT+Softmax+Focal_loss      | 0.23 | 94.23     | 70.42     | 79.92     | 74.87     |
| BERT+Softmax+Label_Smoothing | 0.21 | 94.33     | 72.95     | 80.34     | 76.47     |
| BERT+Softmax+CE+DA_wo_ori    | 0.27 | 93.12     | 73.85     | 70.15     | 71.95     |
| BERT+Softmax+CE+outside      | 0.24 | 94.20     | 74.37     | **80.86** | 77.48 |
| BERT+Softmax+CE+outside+dedup_DA_3    | 0.45 | 93.70     | 74.47     | 78.71     | 76.53     |
| RoBERTa+Softmax+CE+outside   | 0.34 | 94.00     | 74.44     | 79.65     | 76.96     |
| RoBERTa-large+Softmax+CE+outside   | 0.26 |  **94.33**  | **75.73**     | 80.53     |  **78.06**    |

**Result on the dev set (official f1)**

|                              | address           | book       | company       | game          | government               | movie          | name           | organization   | position       | scene  | Macro f1 |
| ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| BERT+Softmax+CE_loss         | 64.18   | 82.89 | 78.18   | 85.96 | 80.78      | 82.43 | 88.86 | 80.56        | 79.20    | 73.76 | 79.63    |
| BERT+Softmax+Focal_loss      |         |       |         |       |            |       |       |              |          |       |          |
| BERT+Softmax+Label_Smoothing |         |       |         |       |            |       |       |              |          |       |          |
| BERT+Softmax+CE+DA_wo_ori    |         |       |         |       |            |       |       |              |          |       |          |
| BERT+Softmax+CE+outside      | 64.25   | 81.85 | 80.80   | 85.71 | 82.51      | 81.31 | 88.86 | 80.34        | 79.58    | 73.50 | 79.87    |
| BERT+Softmax+CE+outside+dedup_DA_3    | 60.23   | 76.22 | 81.69   | 82.69 | 82.28      | 81.79 | 86.16 | 77.87        | 77.79    | 69.47 | 77.62    |
| RoBERTa+Softmax+CE+outside   | 61.83   | 81.08 | 81.73   | 84.81 | 81.89      | 78.77 | 88.07 | 77.12        | 79.39    | 71.03 | 78.57    |
| RoBERTa-large+Softmax+CE+outside   | 63.04   | 83.55 | 83.70   | 85.62 | 82.45      | 85.52 | 89.71 | 81.11        | 79.11    | 73.10 | 80.69    |