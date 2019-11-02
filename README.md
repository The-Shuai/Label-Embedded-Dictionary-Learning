# Label-Embedded-Dictionary-Learning
A class shared dictionary learning method for image classification
Created by Shuai Shao, Rui Xu, Weifeng Liu, Bao-Di Liu, Yan-Jiang Wang from China University of Petroleum.<br>
![image](https://github.com/The-Shuai/Label-Embedded-Dictionary-Learning/blob/master/doc/Comparasion.png)

## Introduction<br>
We propose a novel dictionary learning algorithm named label embedded dictionary learning (LEDL). This method introduces the L1-norm regularization term to replace the L0-norm regularization of LC-KSVD. Compared with L0-norm, the sparsity constraint factor of L1-norm is unfixed so that the basis vectors can be selected freely for linear fitting. Thus, our proposed LEDL method can get smaller errors than LC-KSVD. In addition, L1-norm sparse representation is widely used in many fields so that our proposed LEDL method can be extended and applied easily.  You can also check out [paper](https://arxiv.org/abs/1903.03087) for a deeper introduction.<br>

In this repository, we release the code and data for training and testing.<br>
In our paper, the proposed LEDL is compared with some traditional machine learning [baselines](https://github.com/The-Shuai/Visual-Classifier-Baselines)

## Usage<br>
Modify the number of training set and testing set.<br>

```matlab 
options.tr_num = 10; % randomly select 10 samples per class as the training data
options.val_num = 5; % randomly select 5 samples per class as the training data
```



## License
Our code is released under MIT License (see LICENSE file for details).



