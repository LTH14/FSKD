# Few Sample Knowledge Distillation for Efficient Network Compression

This repository contains the samples code for FSKD, [Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839) (CVPR 2020) by Tianhong Li, Jianguo Li, Zhuang Liu and Changshui Zhang.

The repo shows how to train a VGG-16 model on CIFAR-10 and then prune it with very few unlabeled samples using FSKD. It can also be extended to other models, network pruning / decoupling methods and datasets.
## Training VGG-16

```shell
python main.py --dataset cifar10 --arch vgg --depth 16 --lr 0.01
```

## Prune and FSKD

```shell
python vggprune_pruning.py --dataset cifar10 --depth 16 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT] --num_sample 500
```

## main.py

This file performs training of different network structure. You can specify the training parameter as well as whether using sparsity training or not.

## vggprune_pruning.py

This file performs FSKD on specified VGG pretrained model. You can specify the number of samples used here. The script first build up the pruned VGG model from the original model. It then add the additional 1x1 conv layer, performs FSKD layer by layer (recover_one_layer()), and finally absorb the additional 1x1 conv layer back to the model.

To apply FSKD on your own model and dataset, you may pay more attention on these several functions: add_pwconv() and absorb_pwconv() in models/vgg.py which add and merge the 1x1 conv layer, and recover_one_layer() in vggprune_pruning.py, which estimate the parameters of one 1x1 conv layer using few samples.