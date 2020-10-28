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
| BERT+Softmax+CE_loss         | 0.196659132838 | ==0.94337445== | ==0.734379626== | ==0.807291666== | ==0.7691114901== |
| BERT+Softmax+Focal_loss      | 0.232363879681 | 0.94226025     | 0.704245554     | 0.799153646     | 0.7487038731     |
| BERT+Softmax+Label_Smoothing | 0.210226714611 | 0.94331476     | 0.729530003     | 0.803385417     | 0.7646785438     |

