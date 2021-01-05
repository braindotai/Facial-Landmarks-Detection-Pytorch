# __Face Landmarks Detection__

![Inference on few celebraties](awesome_outputs.webp)

## __Data preprocessing__

1. Augmentations for faces:

    - Random Brightness
    - Random Contrast
    - Random Gamma
    - Random Saturation
    - Random Hue
    - Random Rotation

2. Augmentations for landmarks:

    - Random Rotation

## __Model__

Architecture: [Xception Net](https://arxiv.org/abs/1610.02357)

Summary:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 128, 128]             288
       BatchNorm2d-2         [-1, 32, 128, 128]              64
         LeakyReLU-3         [-1, 32, 128, 128]               0
            Conv2d-4         [-1, 64, 128, 128]          18,432
       BatchNorm2d-5         [-1, 64, 128, 128]             128
         LeakyReLU-6         [-1, 64, 128, 128]               0
            Conv2d-7         [-1, 64, 128, 128]             576
            Conv2d-8         [-1, 64, 128, 128]           4,096
DepthewiseSeperableConv2d-9         [-1, 64, 128, 128]               0
      BatchNorm2d-10         [-1, 64, 128, 128]             128
        LeakyReLU-11         [-1, 64, 128, 128]               0
           Conv2d-12         [-1, 64, 128, 128]             576
           Conv2d-13        [-1, 128, 128, 128]           8,192
DepthewiseSeperableConv2d-14        [-1, 128, 128, 128]               0
      BatchNorm2d-15        [-1, 128, 128, 128]             256
        MaxPool2d-16          [-1, 128, 64, 64]               0
           Conv2d-17          [-1, 128, 64, 64]           8,320
      BatchNorm2d-18          [-1, 128, 64, 64]             256
        LeakyReLU-19          [-1, 128, 64, 64]               0
           Conv2d-20          [-1, 128, 64, 64]           1,152
           Conv2d-21          [-1, 128, 64, 64]          16,384
DepthewiseSeperableConv2d-22          [-1, 128, 64, 64]               0
      BatchNorm2d-23          [-1, 128, 64, 64]             256
        LeakyReLU-24          [-1, 128, 64, 64]               0
           Conv2d-25          [-1, 128, 64, 64]           1,152
           Conv2d-26          [-1, 256, 64, 64]          32,768
DepthewiseSeperableConv2d-27          [-1, 256, 64, 64]               0
      BatchNorm2d-28          [-1, 256, 64, 64]             512
        MaxPool2d-29          [-1, 256, 32, 32]               0
           Conv2d-30          [-1, 256, 32, 32]          33,024
      BatchNorm2d-31          [-1, 256, 32, 32]             512
       EntryBlock-32          [-1, 256, 32, 32]               0
        LeakyReLU-33          [-1, 256, 32, 32]               0
           Conv2d-34          [-1, 256, 32, 32]           2,304
           Conv2d-35          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-36          [-1, 256, 32, 32]               0
      BatchNorm2d-37          [-1, 256, 32, 32]             512
        LeakyReLU-38          [-1, 256, 32, 32]               0
           Conv2d-39          [-1, 256, 32, 32]           2,304
           Conv2d-40          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-41          [-1, 256, 32, 32]               0
      BatchNorm2d-42          [-1, 256, 32, 32]             512
        LeakyReLU-43          [-1, 256, 32, 32]               0
           Conv2d-44          [-1, 256, 32, 32]           2,304
           Conv2d-45          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-46          [-1, 256, 32, 32]               0
      BatchNorm2d-47          [-1, 256, 32, 32]             512
 MiddleBasicBlock-48          [-1, 256, 32, 32]               0
        LeakyReLU-49          [-1, 256, 32, 32]               0
           Conv2d-50          [-1, 256, 32, 32]           2,304
           Conv2d-51          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-52          [-1, 256, 32, 32]               0
      BatchNorm2d-53          [-1, 256, 32, 32]             512
        LeakyReLU-54          [-1, 256, 32, 32]               0
           Conv2d-55          [-1, 256, 32, 32]           2,304
           Conv2d-56          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-57          [-1, 256, 32, 32]               0
      BatchNorm2d-58          [-1, 256, 32, 32]             512
        LeakyReLU-59          [-1, 256, 32, 32]               0
           Conv2d-60          [-1, 256, 32, 32]           2,304
           Conv2d-61          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-62          [-1, 256, 32, 32]               0
      BatchNorm2d-63          [-1, 256, 32, 32]             512
 MiddleBasicBlock-64          [-1, 256, 32, 32]               0
        LeakyReLU-65          [-1, 256, 32, 32]               0
           Conv2d-66          [-1, 256, 32, 32]           2,304
           Conv2d-67          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-68          [-1, 256, 32, 32]               0
      BatchNorm2d-69          [-1, 256, 32, 32]             512
        LeakyReLU-70          [-1, 256, 32, 32]               0
           Conv2d-71          [-1, 256, 32, 32]           2,304
           Conv2d-72          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-73          [-1, 256, 32, 32]               0
      BatchNorm2d-74          [-1, 256, 32, 32]             512
        LeakyReLU-75          [-1, 256, 32, 32]               0
           Conv2d-76          [-1, 256, 32, 32]           2,304
           Conv2d-77          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-78          [-1, 256, 32, 32]               0
      BatchNorm2d-79          [-1, 256, 32, 32]             512
 MiddleBasicBlock-80          [-1, 256, 32, 32]               0
        LeakyReLU-81          [-1, 256, 32, 32]               0
           Conv2d-82          [-1, 256, 32, 32]           2,304
           Conv2d-83          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-84          [-1, 256, 32, 32]               0
      BatchNorm2d-85          [-1, 256, 32, 32]             512
        LeakyReLU-86          [-1, 256, 32, 32]               0
           Conv2d-87          [-1, 256, 32, 32]           2,304
           Conv2d-88          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-89          [-1, 256, 32, 32]               0
      BatchNorm2d-90          [-1, 256, 32, 32]             512
        LeakyReLU-91          [-1, 256, 32, 32]               0
           Conv2d-92          [-1, 256, 32, 32]           2,304
           Conv2d-93          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-94          [-1, 256, 32, 32]               0
      BatchNorm2d-95          [-1, 256, 32, 32]             512
 MiddleBasicBlock-96          [-1, 256, 32, 32]               0
        LeakyReLU-97          [-1, 256, 32, 32]               0
           Conv2d-98          [-1, 256, 32, 32]           2,304
           Conv2d-99          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-100          [-1, 256, 32, 32]               0
     BatchNorm2d-101          [-1, 256, 32, 32]             512
       LeakyReLU-102          [-1, 256, 32, 32]               0
          Conv2d-103          [-1, 256, 32, 32]           2,304
          Conv2d-104          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-105          [-1, 256, 32, 32]               0
     BatchNorm2d-106          [-1, 256, 32, 32]             512
       LeakyReLU-107          [-1, 256, 32, 32]               0
          Conv2d-108          [-1, 256, 32, 32]           2,304
          Conv2d-109          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-110          [-1, 256, 32, 32]               0
     BatchNorm2d-111          [-1, 256, 32, 32]             512
MiddleBasicBlock-112          [-1, 256, 32, 32]               0
       LeakyReLU-113          [-1, 256, 32, 32]               0
          Conv2d-114          [-1, 256, 32, 32]           2,304
          Conv2d-115          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-116          [-1, 256, 32, 32]               0
     BatchNorm2d-117          [-1, 256, 32, 32]             512
       LeakyReLU-118          [-1, 256, 32, 32]               0
          Conv2d-119          [-1, 256, 32, 32]           2,304
          Conv2d-120          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-121          [-1, 256, 32, 32]               0
     BatchNorm2d-122          [-1, 256, 32, 32]             512
       LeakyReLU-123          [-1, 256, 32, 32]               0
          Conv2d-124          [-1, 256, 32, 32]           2,304
          Conv2d-125          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-126          [-1, 256, 32, 32]               0
     BatchNorm2d-127          [-1, 256, 32, 32]             512
MiddleBasicBlock-128          [-1, 256, 32, 32]               0
     MiddleBlock-129          [-1, 256, 32, 32]               0
          Conv2d-130          [-1, 512, 16, 16]         131,584
     BatchNorm2d-131          [-1, 512, 16, 16]           1,024
       LeakyReLU-132          [-1, 256, 32, 32]               0
          Conv2d-133          [-1, 256, 32, 32]           2,304
          Conv2d-134          [-1, 256, 32, 32]          65,536
DepthewiseSeperableConv2d-135          [-1, 256, 32, 32]               0
     BatchNorm2d-136          [-1, 256, 32, 32]             512
       LeakyReLU-137          [-1, 256, 32, 32]               0
          Conv2d-138          [-1, 256, 32, 32]           2,304
          Conv2d-139          [-1, 512, 32, 32]         131,072
DepthewiseSeperableConv2d-140          [-1, 512, 32, 32]               0
     BatchNorm2d-141          [-1, 512, 32, 32]           1,024
       MaxPool2d-142          [-1, 512, 16, 16]               0
          Conv2d-143          [-1, 512, 16, 16]           4,608
          Conv2d-144          [-1, 512, 16, 16]         262,144
DepthewiseSeperableConv2d-145          [-1, 512, 16, 16]               0
     BatchNorm2d-146          [-1, 512, 16, 16]           1,024
       LeakyReLU-147          [-1, 512, 16, 16]               0
          Conv2d-148          [-1, 512, 16, 16]           4,608
          Conv2d-149         [-1, 1024, 16, 16]         524,288
DepthewiseSeperableConv2d-150         [-1, 1024, 16, 16]               0
     BatchNorm2d-151         [-1, 1024, 16, 16]           2,048
       LeakyReLU-152         [-1, 1024, 16, 16]               0
AdaptiveAvgPool2d-153           [-1, 1024, 1, 1]               0
         Dropout-154           [-1, 1024, 1, 1]               0
       ExitBlock-155           [-1, 1024, 1, 1]               0
          Linear-156                  [-1, 136]         139,400
================================================================
Total params: 2,630,888
Trainable params: 2,630,888
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.06
Forward/backward pass size (MB): 441.02
Params size (MB): 10.04
Estimated Total Size (MB): 451.12
----------------------------------------------------------------
```

## __Training hyperparameters__

1. Objective loss : `MSELoss`
2. Optimizer      : `Adam`
3. Learning Rate  : 0.0008
4. Epochs         : 30

## __Training progress__

![Progress of the model thoughout the training](progress.gif)

## Final Results
![Final outputs of the model](output.gif)

## [__Read Full Article__](https://medium.com/swlh/facial-landmarks-detection-using-xception-net-908b8b80f758)

## __Watch the video__

[![Watch the video here](https://img.youtube.com/vi/Q8oJxOSRMSw/0.jpg)](https://www.youtube.com/watch?v=Q8oJxOSRMSw)

# Author - Rishik Mourya
