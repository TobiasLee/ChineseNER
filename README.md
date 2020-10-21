# ChineseNER



## Prepare environment
'''bash
conda create -n ner python=3.6
conda activate ner
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
'''

## download pre-trained Chinese BERT

> sh download.sh 

## Run

> sh ner.sh 
