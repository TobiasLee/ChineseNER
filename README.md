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

# Run

> sh ner.sh 


# Result
eval_loss = 0.1966591328382492
eval_accuracy_score = 0.9433744528452049
eval_precision = 0.7343796268877703
eval_recall = 0.8072916666666666
eval_f1 = 0.7691114901535123
epoch = 3.0
total_flos = 2518344871345152
