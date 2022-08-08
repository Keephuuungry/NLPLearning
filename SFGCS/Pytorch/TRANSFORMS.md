# TRANSFORMS

## transforms.Compose

`torchvision.transforms`是pytorch的图像预处理包，一般会用`transforms.Compose`将多个处理步骤整合到一起，支持链式处理。如：

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),# 缩放
    transforms.RandomCrop(32, padding=4), #裁剪
    transforms.ToTensor(), # 转为张量，同时归一化
    transforms.Normalize(norm_mean, norm_std),# 标准化
])
```

然后再通过自定义`Dataset`类中的`__getitem__`函数，进行图片的预处理。

## transforms的脚本化

在torch1.7新增了该特性，现在的transform支持以下方式：

```python
import torch
import torchvision.transforms as T

# to fix random seed, use torch.manual_seed
# instead of random.seed
torch.manual_seed(12)

transforms = torch.nn.Sequential(
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(transforms)

tensor_image = torch.randint(0, 256, size=(3, 256, 256), dtype=torch.uint8)
# works directly on Tensors
out_image1 = transforms(tensor_image)
# on the GPU
out_image1_cuda = transforms(tensor_image.cuda())
# with batches
batched_image = torch.randint(0, 256, size=(4, 3, 256, 256), dtype=torch.uint8)
out_image_batched = transforms(batched_image)
# and has torchscript support
out_image2 = scripted_transforms(tensor_image)
```

transforms脚本化的优点：

- 数据增强可以支持GPU加速
- Batch化transformation，视频处理中使用更方便
- 可以支持多channel的tensor增强，而不仅仅是3通道或者4通道的tensor

## 其他预处理函数

- `Resize`：把给定的图片resize到指定大小
- `Normalize`：对图像进行标准化
- `ToTensor`：将numpy.ndarray或PIL数据类型转换成Tensor数据类型
- `CenterCop`：中心剪切
- `ColorJitter`：随机改变图像的亮度对比度饱和度
- `ToPILImage`：将tensor转换为PIL图像
- `RandomCrop`：在一个随机位置进行裁剪
- `RandomHorizontalFlip`：以0.5的概率水平翻转给定的PIL图像
- `RandomVerticalFlip`：以0.5的概率竖直翻转给定的PIL图像
- `RandomResizedCrop`：将PIL图像裁剪成任意大小和横纵比
- `Grayscale`：将图像转换为灰度图像
- `RandomGrayscale`：将图像以一定的概率转换为灰度图像
- `FiceCrop`：把图像裁剪为四个角和一个中心

