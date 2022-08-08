# Pytorch

- `torch.cuda.is_available() `：查看torch是否可

- `dir()`：打开，看见某工具包

- `help()`：查看函数的使用说明（去掉函数的括号）

  函数加括号表示引用该函数；不加括号表示函数本身

---

Dataset：数据集

DataLoader：将数据集打包传入模型的工具

```python
///P7:Dataset类代码实战///
from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "dataset/train"
label_dir = "ants"
dataset = Mydata(root_dir, label_dir)
img, label = dataset[0]
```

其他：

图像处理库PIL：`from PIL import Image`

`os.listdir()`：将路径下的文件名，整合到一个列表中

`os.path.join()`：将两个路径进行拼接



---

## ※Tensorboard

导入：`from torch.utils.tensorboard import SummaryWritter`