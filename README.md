# Keras classifier (with converting h5 to tflite)

## 1. setting  
    pip install -r requirements.txt  

    You can choice cifar10 datasets, if you want to train classification model quickly because it is small. 
    ImageNet, on the other hand, it is a much larger dataset. 
    ImageNet has become a standard benchmark dataset for evaluating the performance of image classification.

    python -m prepare_dataset --name tinyimagenet --extract True --output ./dataset  

    python -m prepare_dataset --name cifar --extract True --output ./dataset  

    python -m prepare_dataset --name mnist --extract True --output ./dataset   

## 2. Train  
    python -m train --name resnet18 --dataset tinyimagenet --epochs 300 --batch 128 --logdir ./logs --weight_decay 0.0005 --save True  
    python -m train --name resnet18 --dataset cifar --epochs 300 --batch 128 --logdir ./logs --weight_decay 0.0005 --save True  



