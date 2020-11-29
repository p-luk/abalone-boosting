# abalone-boosting

### Dataset
The [Abalone dataset](http://archive.ics.uci.edu/ml/datasets/Abalone) is not included.

### Algorithm

An implementation of AdaBoost for Mohri's Foundations of Machine Learning class. The original algorithm is implemented according to the below pseudocode:

<img src="https://github.com/p-luk/abalone-boosting/blob/main/adaboost_pseudocode.png" width="600">

[Source](https://cs.nyu.edu/~mohri/mls/ml_boosting.pdf)

Vanilla AdaBoost uses loss function `exploss` defined by <img src="https://render.githubusercontent.com/render/math?math=\Phi(u) = e^{-u}">. In addition, we define`logisticloss` a custom loss function <img src="https://render.githubusercontent.com/render/math?math=\Phi(u) = \log_2(1+e^{-u})">
