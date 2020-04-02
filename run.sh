# Script for running baseline (cross-entropy)

CUDA_VISIBLE_DEVICES=0 python train.py --no_over --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'ERM' --warm 200 --epoch 200  

# Script for running over-sampling

CUDA_VISIBLE_DEVICES=0 python train.py --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'Over' --warm 0 --epoch 200  

# SMOTE

CUDA_VISIBLE_DEVICES=0 python train.py -s --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'SMOTE' --warm 0 --epoch 200  

# Script for running re-weighting (RW)

CUDA_VISIBLE_DEVICES=0 python train.py --no_over -c --eff_beta 1.0 --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'Cost' --warm 0 --epoch 200  

# Script for running re-weighting with class-balanced loss (CB-RW)

CUDA_VISIBLE_DEVICES=0 python train.py --no_over -c --eff_beta 0.999 --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'CBLoss' --warm 0 --epoch 200  

# Script for running DRS

CUDA_VISIBLE_DEVICES=0 python train.py --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'DRS' --warm 160 --epoch 200

# Script for running Focal

CUDA_VISIBLE_DEVICES=0 python train.py --no_over --loss_type Focal --focal_gamma 1.0 --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'Focal' --warm 160 --epoch 200  

# Script for running LDAM-DRW

CUDA_VISIBLE_DEVICES=0 python train.py --no_over -c --loss_type LDAM --eff_beta 0.999 --ratio 100 --decay 2e-4 --model resnet32_norm --dataset cifar10 --lr 0.1 --batch-size 128 --name 'LDAM-DRW' --warm 160 --epoch 200  

# Script for running our method (M2m) 
CUDA_VISIBLE_DEVICES=0 python train.py -gen --ratio 100 --decay 2e-4 --model resnet32 --dataset cifar10 --lr 0.1 --batch-size 128 --name 'M2m' --beta 0.999 --lam 0.5 --gamma 0.9 --step_size 0.1 --attack_iter 10 --warm 160 --epoch 200 --net_g ./checkpoint/erm_r100_c10_trial1.t7 

