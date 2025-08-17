
# STOP/OSSTOP
the code is based on spikingjelly.activation_based


1. Dataset+Network_Architecture

| Dataset    | Network_Architecture |
|------------|----------------------|
| CIFAR10    | VGG11_BN |
| CIFAR100   | ResNet-18             |
| DVSCIFAR10 | VGG11_BN             |
| DVSGesture | VGG11_BN             |
for VGG11_BN, you need to make sure args.bn is True

3. you can control the learning method by modify the args.os as follows:

| Method    | args.os |
|-----------|---------|
| STOP      | False   |
| OSSTOP    | True    |

4. Comparisons of Accuracy ,Speed, Computation resources: 
    the number means the rank of the performance of the specific aspect

| Method    | Accuracy | Speed | computation resources |
|-----------|----------|-------|-----------------------|
| STOP      | 1        | 2     | 2                     |
| OSSTOP    | 2        | 1     | 1                     |

5. we assume all the network operate without bias(including Conv2d and Linear) 

8. This repository contains a PyTorch implementation for the STOP and its variants

### Dependencies
- torch v1.0.1
- torchvision v0.2.1




