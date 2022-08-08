# Pytorch Q & A

[TOC]



### Q1：`torch.nn`与`torch.nn.functional`的区别与联系

发现`nn.Conv2d()`和`nn.functional.conv2d()`都可以用来构建模型，区别与联系？

> 参考网址：https://www.zhihu.com/question/66782101

#### 相同之处

- `nn.Xxx`和`nn.funcitonal.xxx`实际功能是相同的，即`nn.Conv2d`和`nn.functional.conv2d`都是进行卷积，`nn.Dropout`和`nn.functional.dropout`都是进行dropout。
- 运行效率近乎相同。

`nn.functional.xxx`是函数接口，而`nn.Xxx`是`nn.functional.xxx`的类封装，并且`nn.Xxx`都继承于一个共同祖先`nn.Module`。这一点导致`nn.Xxx`除了具有`nn.functional.xxx`功能之外，内部附带了`nn.Module`相关的属性和方法，例如`train(), eval(),load_state_dict, state_dict `等。

#### 差别之处

- 两者的调用方式不同

  `nn.Xxx`需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

  ```python
  inputs = torch.rand(64, 3, 244, 244)
  conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
  out = conv(inputs)
  ```

  `nn.functional.xxx`同时传入输入数据和weight,bias等其他参数

  ```python
  weight = torch.rand(64, 3, 3, 3)
  bias = torch.rand(64)
  out = nn.functional.conv2d(inputs, weight, bias, padding=1)
  ```

- `nn.Xxx`继承于`nn.Module`，能够很好的与`nn.Sequential`结合使用，而`nn.functional.xxx`无法与`nn.Sequential`结合使用。

  ```python
  fm_layer = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(0.2)
  )
  ```

- `nn.Xxx`不需要自己定义和管理weight；而`nn.functional.xxx`需要自己定义weight，每次调用时都需要手动传入weight，不利于代码复用。

  使用`nn.Xxx`定义一个CNN

  ```python
  class CNN(nn.Module):
      def __init__(self):
          super(CNN, self.__init__()
          self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=0)
          self.relu1 = nn.ReLU()
          self.maxpool1 = nn.Maxpool2d(kernel_size=2)
                
          self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0)
          self.relu2 = nn.ReLU()
          self.maxpool2 = nn.Maxpool2d(kernel_size=2)
          
         	self.linear1 = nn.Linear(4*4*32, 10)
      def forward(self, x):
         	x = x.view(x.size(0), -1)
       	out = self.maxpool1(self.relu1(self.cnn1(x)))
          out = self.maxpool2(self.relu2(self.cnn2(out)))
          out = self.linear2(out.view(x.size(0), -1))
          return out
  ```

  使用`nn.functional.xxx`定义与上面相同的CNN

  ```python
  class CNN(nn.Module):
      def __init__(self):
          super(CNN, self).__init__()
          
          self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))
          self.bias1_weight = nn.Parameter(torch.rand(16))
          
          self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
          self.bias2_weight = nn.Parameter(torch.rand(32))
          
          self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
          self.bias3_weight = nn.Parameter(torch.rand(10))
          
      def forward(self, x):
          x = x.view(x.size(0), -1)
          out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
          out = F.relu(out)
          out = F.max_pool2d(out)
          
          out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
          out = F.relu(out)
          out = F.max_pool2d(out)
          
          out = F.linear(x, self.linear1_weight, self.bias3_weight)
          return out
  ```

  > 好像共享参数的话，后者方便？

官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。

关于dropout，知乎作者强烈推荐使用`nn.Xxx`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用`nn.Xxx`方式定义dropout，在调用`model.eval()`之后，model中所有的dropout layer都关闭，但以`nn.function.dropout`方式定义dropout，在调用`model.eval()`之后并不能关闭dropout。

### Q2：`model.named_parameters()`与`model.parameters()`

区别：

- `named_parameters()`：`for name, param in model.named_parameters():`可以访问`name`属性。

- `parameters()`：`for index, param in enumerate(model.parameters()):`

联系：

均可以更改参数的可训练属性`param.requires_grad`。

### Q3：Logits的含义

原公式：$logit(p)=log\frac{p}{1-p}$，Logti=Logistic Unit

logtis可以看作神经网络输出的**未经过归一化(Softmax等）的概率**，一般是全连接层的输出。将其结果用于分类任务计算loss时，如求cross_entropy的loss函数会设置from_logits参数。

因此，当from_logtis=False（默认情况）时，可以理解为输入的y_pre不是来自logtis，那么它就是已经经过softmax或者sigmoid归一化后的结果了；当from_logtis=True时，则可以理解为输入的y_pre是logits，那么需要对它进行softmax或者sigmoid处理再计算cross entropy。

```python
y_pre = np.array([[5, 5], [2, 8]], dtype=np.float)   # 神经网络的输出，未概率归一化
y_true = np.array([[1, 0], [0, 1]], dtype=np.float)
loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pre, from_logits=True)
print(loss)

y_pre = tf.nn.sigmoid(y_pre)
loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pre, from_logits=False)
print(loss)
```

### Q4：`requires_grad`、`grad_fn`、`with torch.no_grad()`、`retain_grad`

- `requires_grad`是pytorch中tensor的一个属性，如果`requires_grad=True`，在进行反向传播的时候会记录该tensor梯度信息。

- `grad_fn`也是tensor的一个属性，它记录的是tensor的运算信息，如c=a+b，那么`c.grad_fn=<AddBackward0>`。

- `with torch.no_grad():`，在该模块下，所有计算得出的tensor的`requires_grad`都自动设置为False，不参与反向传播。

- `retain_graph`：如果设置为`False`，计算图中的中间变量在计算完后就会被释放。Pytorch的机制是每次调用`.backward()`都会free掉所有buffers，模型中可能有多次`backward()`，而前一次`backward()`存储在buffer中的梯度，会因为后一次调用`backward()`被free掉。

  ==注==：

  关于`requires_grad`和`torch.no_grad()`，有两篇文章写的很好，下附链接。大意是：`with torch.no_grad()`时会**阻断该传播路线的梯度传播**，而`requires_grad`置`False`时，**不会阻断该路线的梯度传播，只是不会计算该参数的梯度**。

  在使用`with torch.no_grad()`时，虽然可以少计算一些tensor的梯度而减少计算负担，但是如果有`backward`的时候，很可能会有错误的地方，要么很确定没有`backward`就可以用，要么在显卡允许的情况下就不用`with torch.no_grad()`，以免出现不必要的错误。

> requires_grad与torch.no_grad()：
>
> 1. https://its201.com/article/laizi_laizi/112711521
> 2. https://www.cxyzjd.com/article/weixin_43145941/114757673
>
> retain_graph：
>
> 1. https://www.cnblogs.com/luckyscarlett/p/10555632.html



