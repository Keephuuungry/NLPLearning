# DATASETS & DATALOADERS

Pytorch提供了两个原始接口用于处理和数据相关的任务，分别是`torch.utils.data.DataLoader` 和`torch.utils.data.Dataset`。Dataset用于储存样本和对应的标签信息，Dataloader用迭代器的方式封装了Dataset。

Pytorch提供了三个特定领域的库[`TorchText`](https://link.zhihu.com/?target=https%3A//pytorch.org/text/stable/index.html)，[`TorchVision`](https://link.zhihu.com/?target=https%3A//pytorch.org/vision/stable/index.html)和 [`TorchAudio`](https://link.zhihu.com/?target=https%3A//pytorch.org/audio/stable/index.html)，分别包含了文本数据，图像数据和音频数据，每个库都内置了几十种经典的各自领域的数据样本。调用：`torchvision.datasets`、`torchtext.datasets`、`torchaudio.datasets`。

>`torchvision`计算机视觉工具包，有三个主要的模块
>
>- `torchvision.transforms`：包括常用的图像预处理方法
>- `torchvision.datasets`：包括常用的数据集，如mnist、CIFAR-10、Image-Net等
>- `torchvision.models`：包括常用的预训练好的模型，如AlexNet、VGG等

在使用Pytorch数据准备时分为两步：

- `datasets`API下载数据/使用自带的数据集
- `DataLoader`API迭代数据

## `datasets`API下载数据 / 使用自带的数据集

1. 使用Pytorch`Dataset`库中提供的数据集

   ```python
   import torch
   from torch.utils.data import Dataset
   from torchvision import datasets
   from torchvision.io import ToTensor
   
   #以FashionMNIST举例
   train_data = datasets.FashionMNIST(
       root="path",
       train=True,
       download=True,
       transform=ToTensor()
   )
   
   test_data = datasets.FashionMNIST(
       root="data",
       train=False,
       download=True,
       transform=ToTensor()
   )
   ```

   - `root` is the path where the train/test data is stored.

   - `train` specifies training or test dataset.

   - `download=True` downloads the data from the internet if it's not available at `root`.

   - `transform` and `target_transform` specify the feature and label transformations.

2. 使用自带的数据集

   这类情况需要我们定义自己的Dataset类来实现灵活的数据读取，定义的类需要继承Pytorch自身的Dataset类，主要包含三个函数：

   - `__init__`：用于向类中传入外部参数，同时定义样本集
   - `__getitem__`：用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
   - `__len__`：用于返回数据集的样本数

   ```python
   import os
   import pandas as pd
   from torchvision.io import read_image
   from torch.utils.data import Dataset
   """
   例：将图片存储在image_path目录下，标签以CSV文件形式存储在label_csv_path目录下。
   
   label_csv格式如下：
       tshirt1.jpg, 0
       tshirt2.jpg, 0
       ......
   	ankleboot999.jpg, 9
   
   如果数据集图片-标签的存储形式不同，相应的代码也需要修改。
   最终目的是通过每张图片的路径读取图片，将image及对应的label返回。
   """
   class CustomImageDataset(Dataset):
   	def __init__(self, image_path, label_csv_path, transform=None, target_transform=None):
           self.img_path = image_path
       	self.label_csv = pd.read_csv(label_csv_path)
           self.transform = transform
           self.target_transform = target_transform
           
       def __len__(self):
           return len(self.label_csv)
       
       def __getitem__(self, idx):
           img_path = os.path.join(self.img_path, self.label_csv.iloc[idx,0])
           image = read_image(img_path)
           label = self.label_csv.iloc[idx,1]
           if self.transform:
               image = self.transform(image)
           if self.target_transform:
               image = self.target_transform(image)
           return image, label
   ```

   > 可以用不同的工具包读取数据：
   >
   > - ```python
   >   from PIL import Image
   >   image = Image.open("file_path")
   >   # 读取的image数据类型为PIL
   >   ```
   >
   > - ```python
   >   from torchvision.io import read_image
   >   image = read_image("file_path")
   >   # 读取的image数据类型为Tensor
   >   ```
   >
   > - ```python
   >   # opencv库
   >   import cv2
   >   image = cv2.imread("file_path")
   >   # 读取的image数据类型为numpy.ndarray
   >   ```

   ## `DataLoader`API迭代数据

   构建好Dataset后，就可以使用DataLoader来按批次读入数据了，实现代码如下：

   ```python
   from torch.utils.data import DataLoader
   
   train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
   ```

   `DataLoader`函数常用参数：

   - `batch_size`：每次读入的样本数
   - `num_workers`：有多少个进程用于读取数据
   - `shuffle`：是否将读入的数据打乱
   - `drop_last`：对于样本最后一部分没有达到批次数的样本，使其不再参与训练

Pytorch中的`DataLoader`读取可以使用`next`和`iter`来完成：

```python
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
plt.show()
```

