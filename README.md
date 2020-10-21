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
eval_loss = 0.196659132838

eval_accuracy_score = 0.94337445

eval_precision = 0.734379626

eval_recall = 0.807291666

eval_f1 = 0.7691114901
