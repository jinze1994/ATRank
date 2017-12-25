# ATRank
An Attention-Based User Behavior Modeling Framework for Recommendation

## Introduction
This is an implementation of the paper [ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation.](https://arxiv.org/abs/1711.06632) Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, Jun Gao. AAAI 2018.

Bibtex:
```sh
@article{zhou2018ATRank,
  title={ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation},
  author={Zhou, Chang and Bai, Jinze and Song, Junshuai and Liu, Xiaofei and Zhao, Zhengchao and Chen, Xiusi and Gao, Jun},
  journal={arXiv preprint arXiv:1711.06632},
  year={2017}
}
```

This repository also contains all the competitor's methods mentioned in the paper. Some implementations consults the [Transfomer](https://github.com/Kyubyong/transformer), and [Text-CNN](https://github.com/dennybritz/cnn-text-classification-tf).

Note that, the heterogeneous behavior datasets used in the paper is private, so you could not run multi-behavior code directly.
But you could run the code on amazon dataset directly and review the heterogeneous behavior code.

## Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* Pandas >= 0.20.1
* TensorFlow >= 1.4.0 (Probably earlier version should work too, though I didn't test it)
* GPU with memory >= 10G

## Download dataset and preprocess
* Step 1: Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
```sh
mkdir raw_data/;
cd utils;
bash 0_download_raw.sh;
```
* Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```sh
python 1_convert_pd.py;
python 2_remap_id.py
```

## Training and Evaluation
This implementation not only contains the ATRank method, but also provides all the competitors' method, including BPR, CNN, RNN and RNN+Attention. The training procedures of all method is as follows:
* Step 1: Choose a method and enter the folder.
```
cd atrank;
```
Alternatively, you could also run other competitors's methods directly by `cd bpr` `cd cnn` `cd rnn` `cd rnn_att`,
and follow the same instructions below.

Note that, the heterogeneous behavior datasets used in the paper is private, so you could't run the code of this part directly.
But you could review the neural network code we use in this paper by `cd multi`.
* Step 2: Building the dataset adapted to current method.
```
python build_dataset.py
```
* Step 3: Start training and evaluating using default arguments in background mode. 
```
python train.py >log.txt 2>&1 &
```
* Step 4: Check training and evaluating progress.
```
tail -f log.txt
tensorboard --logdir=save_path
```
Note that the evaluating producure alternate with training producure, so run the command above may cost five to ten hours until converge completely according to the different methods. If you need to kill that job instantly:
```
nvidia-smi  # Fetch the PID of current training process.
kill -9 PID # Kill the target process.
```

You could change the training and networks hyperparameters by command arguments, like `python train.py --learning_rate=0.1`. To see all command arguments you could use `python train.py --help`.

## Results
You always could use `tensorboard --logdir=save_path` to see the AUC curve and check all kinds of embedding histogram.
The collected AUC curve of test set is as follows

<img src="https://github.com/jinze1994/ATRank/blob/master/utils/auc.png" height = "250" alt="AUC curve in test set" align=left />
