## pickle

##### 常用函数

pickle.dumps(obj)：将obj对象序列化为string形式

pickle.loads(string)：从string中读出序列化前的obj对象

```python
>>> import pickle
>>> li = [1, 2, 3]
>>> pickle_dumps = pickle.dumps(li) #把 li 对象序列化
>>> pickle_dumps
b'\x80\x03]q\x00(K\x01K\x02K\x03e.'

>>> pickle_loads = pickle.loads(pickle_dumps) #转换为序列化之前的数据
>>> pickle_loads
[1, 2, 3]
```

pickle.dump(obj, file)：将obj对象序列化存入已经打开的file中

pickle.load(file)：将file中的对象序列化读出

```python
>>> import pickle
>>> li = [1, 2, 3]
>>> pickle_dumps = pickle.dumps(li)

#将序列化后的对象存入文件
>>> with open('pickle_test', 'wb') as f:
   		#f.write(pickle_dumps)  
...     pickle.dump(li, f) #和上一句f.write(pickle_dumps)效果一样
...


>>> with open('pickle_test', 'rb') as f:
   		#pickle.loads(f.read())
...     pickle.load(f)  #和上一句pickle.loads(f.read())效果一样
...
[1, 2, 3]
```

**注意区别dumps与dump/loads与load**

