# Label-Embedded-Dictionary-Learning
A class shared dictionary learning method for image classification, has been published in Neurocomputing, 2019.
Created by Shuai Shao, Rui Xu, Weifeng Liu, Bao-Di Liu, Yan-Jiang Wang from China University of Petroleum.<br>


## Introduction<br>
We propose a novel dictionary learning algorithm named label embedded dictionary learning (LEDL). This method introduces the L1-norm regularization term to replace the L0-norm regularization of LC-KSVD. Compared with L0-norm, the sparsity constraint factor of L1-norm is unfixed so that the basis vectors can be selected freely for linear fitting. Thus, our proposed LEDL method can get smaller errors than LC-KSVD. In addition, L1-norm sparse representation is widely used in many fields so that our proposed LEDL method can be extended and applied easily.  You can also check out [paper](https://sci-hub.do/10.1016/j.neucom.2019.12.071) for a deeper introduction.<br>

In this repository, we release the code and data for training and testing.<br>
In our paper, the proposed LEDL is compared with some traditional machine learning [baselines](https://github.com/The-Shuai/Visual-Classifier-Baselines)

## Citation
if you find our work useful in your research, please consider citing:<br>
```
@article{shao2020label,
  title={Label embedded dictionary learning for image classification},
  author={Shao, Shuai and Xu, Rui and Liu, Weifeng and Liu, Bao-Di and Wang, Yan-Jiang},
  journal={Neurocomputing},
  volume={385},
  pages={122--131},
  year={2020},
  publisher={Elsevier}
}
```

## Usage<br>
Firstly, you should download the [datasets](https://pan.baidu.com/s/1zEDDzRB2Dbz_otDWUMflMQ)

Then, modify the number of training set and testing set.<br>

```matlab 
options.tr_num = 10; % randomly select 10 samples per class as the training data
options.val_num = 5; % randomly select 5 samples per class as the training data
```
## License
Our code is released under MIT License (see LICENSE file for details).



