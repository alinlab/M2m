# M2m: Imbalanced Classification via Major-to-minor Translation

This project if for the paper 
**M2m: Imbalanced Classification via Major-to-minor Translation**.
## Dependencies
```
pip install tqdm
```
## Scripts
### Training Scripts
Now Testing
```
# CUDA_VISIBLE_DEVICES=0 python train.py -r -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 --step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_t ./checkpoint/ckpt_erm_epoch159_919.t7 --net_g ./checkpoint/erm_r100_c10_trial1.t7 
```
### Testing Scripts
Now Testing
