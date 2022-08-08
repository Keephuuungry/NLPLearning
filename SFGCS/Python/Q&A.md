# Python

### Q1：函数可变参数`*args` 和 `**kwargs`

> 参考网站：http://showteeth.tech/posts/38814.html

这两个参数都是表示给函数传不定数量的参数（不确定最后使用这个函数的时候会传递多少参数，也叫可变参数），两者的差异主要是在：

- `*args`：接受不定量的非关键字参数，如`test('Hello','Welcome')`
- `**kwargs`：接受不定量个关键字参数，如`test(x=1,y=2)`

`*args`和`kwargs`参数关键的是最前面的`*`和`**`，至于后面的字母`args`和`kwargs`只是约定俗称的叫法。

此外，可变参数在函数调用时一定在普通参数之后，如果调换顺序会报错。

### Q2：`super().__init__()`

`super().__init__()`，就是继承父类的`init`方法。

python`class`继承时，只能继承父类中的函数，而不能继承父类中的属性。

```markdown
class Root(object):
  def __init__(self):
      self.x= '这是属性'

  def fun(self):
  	#print(self.x)
      print('这是方法')
      
class A(Root):
  def __init__(self):
      print('实例化时执行')

test = A()		#实例化类
test.fun()	#调用方法
test.x		#调用属性
```

输出：

```markdown
Traceback (most recent call last):

实例化时执行

这是方法

  File "/hom/PycharmProjects/untitled/super.py", line 17, in <module>

    test.x  # 调用属性

AttributeError: 'A' object has no attribute 'x'
```

可以看到，此时父类的方法继承成功，但是父类的属性并未继承。

```markdown
class Root(object):
  def __init__(self):
      self.x = '这是属性'

  def fun(self):
      print(self.x)
      print('这是方法')


class A(Root):
  def __init__(self):
      super(A,self).__init__()
      print('实例化时执行')


test = A()  # 实例化类
test.fun()  # 调用方法
test.x  # 调用属性
```

输出：

```markdown
实例化时执行

这是属性

这是方法
```

==注==：`super()`在python2/3中的区别：

- Python3可以直接写成`super().方法名(参数)`
- Python2必须写成`super(父类,self).方法名(参数)`