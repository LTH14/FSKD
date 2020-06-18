Here is the sample code to reproduce the FSKD result on network pruning, vgg16.
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

## models/vgg.py

This file contains network structure of VGG. Besides the standard structure, the VGG class also contains two additional function: add_pwconv() and absorb_pwconv(), which add and absorb the 1x1 conv respectively.