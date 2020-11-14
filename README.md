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

## Run
> sh ner.sh 

## Result
**Result on the dev set**

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

