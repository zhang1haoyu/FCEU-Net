# FCEU-Net
submitted to [The Visual Computer](https://link.springer.com/journal/371)
![structure](https://github.com/zhang1haoyu/FCEU-Net/blob/main/img/framework.png)

# Abstract
In the realm of remote sensing image (RSI) semantic segmentation, achieving high accuracy remains challenging due to imbalanced class distributions and high similarity among certain categories. This paper introduces an innovative model, named FCEU-Net, which enhances key feature representation while suppressing irrelevant features through three pivotal modules: the Fusion Block, UpResBlock, and Edge Loss. Specifically，the Fusion Block reduces information redundancy and enhances feature capture flexibility, while the UpResBlock preserves multi-scale features through feature concatenation. The Edge Loss is specifically designed to emphasize edge regions within the image. Experimental results on the ISPRS Vaihingen and Potsdam datasets demonstrate that FCEU-Net achieves an MIoU of 73.16% and 77.25%, with m-F1 score of 84.20% and 87.01%, respectively, confirming its effectiveness for accurate RSI semantic segmentation. The code is available at https://github.com/zhang1haoyu/FCEU-Net

# Prerequisite
+ install dependencies
  
`pip install -r requirements.txt`

+ Download dataset

Download [the Potsdam Datasets](https://pan.baidu.com/s/13rdBXUN_ZdelWNlQZ3Y1TQ?pwd=6c3y)

Download [the Vaihingen Datasets](https://pan.baidu.com/s/1EShNi22VfuIu3e6VygMb8g?pwd=3gsr)

+ Download weights

Download [our trained weights](https://github.com/zhang1haoyu/FCEU-Net/blob/main/7713%20ours.pth)

# Usage
+ Split the datasets

`python util/Potsdam.py`

`python util/Vaihingen.py`

+ Train the network

`python train_Potsdam.py`

`python train_Vaihingen.py`
