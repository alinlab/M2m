# M2m: Imbalanced Classification via Major-to-minor Translation

This repository contains code for the paper
**"M2m: Imbalanced Classification via Major-to-minor Translation"** 
by Jaehyung Kim*, [Jongheon Jeong](https://sites.google.com/view/jongheonj)* and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html). 

## Dependencies

* `python3`
* `pytorch >= 1.1.0`
* `torchvision`
* `tqdm`

## Scripts
Please check out `run.sh` for all the scripts to reproduce the CIFAR-10-LT results reported.

### Training procedure of M2m 
1. Train a baseline network g for generating minority samples
```
python train.py --no_over --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 \
--lr 0.1 --batch-size 128 --name 'ERM' --warm 200 --epoch 200   
```
2. Train another network f using M2m with the pre-trained g
```
python train.py -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 \
--lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 \
--step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_g ./checkpoint/pre_trained_g.t7 
```
We also provide a pre-trained ResNet-32 model of g at `checkpoint/erm_r100_c10_trial1.t7`, 
so one can directly use M2m without pre-training as follows:
```
python train.py -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 \
--lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 \
--step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_g ./checkpoint/erm_r100_c10_trial1.t7
```
