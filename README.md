# M2m: Imbalanced Classification via Major-to-minor Translation

This project is for the paper 
**M2m: Imbalanced Classification via Major-to-minor Translation (CVPR 20)**. Some codes are from [LDAM-DRW](https://github.com/kaidic/LDAM-DRW, "LDAM link")
## Preliminaries
It is tested under Ubuntu Linux 16.04 and Python 3.7 environment, and require Pytorch package to be installed:
* [Pytorch (>= 1.1.0)](https://pytorch.org/, "pytorch link")
* [tqdm](https://github.com/tqdm/tqdm, "tqdm link") 
## Scripts
In ```run.sh```, scripts for all baselines and our method are described under CIFAR-10-LT. Please check it carefully. 
### Training procedure of M2m 
1. Training a baseline network g for the generation 
```
python train.py --no_over --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'ERM' --warm 200 --epoch 200   
```
2. Training a desired network f using a previously trained network g
```
python train.py -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 --step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_g ./checkpoint/pre_trained_g.t7 
```
For easy application, pre-trained models f and g are given as ```./checkpoint/ckpt_erm_epoch159_919.t7``` and ```./checkpoint/erm_r100_c10_trial1.t7```, respectively. With them, M2m can be applied as follow:
```
python train.py -r -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 --step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_t ./checkpoint/ckpt_erm_epoch159_919.t7 --net_g ./checkpoint/erm_r100_c10_trial1.t7  
```
