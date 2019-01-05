### 我这也是借鉴别的写的，有啥不对的请联系我：

邮箱：admihao@163.com

![img](https://raw.githubusercontent.com/BinHaoWang/PM/master/mail-classification/doc/%E5%9E%83%E5%9C%BE%E9%82%AE%E7%AE%B1.png?raw=true)


-- -


![img](https://raw.githubusercontent.com/BinHaoWang/PM/master/mail-classification/doc/%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.jpg?raw=true)


[参考网址](https://zhuanlan.zhihu.com/p/35944222)

- text_cnn.py：网络结构设计

- train.py：网络训练

- eval.py：预测&评估

- data_helpers.py：数据预处理


训练：python train.py

测试： python eval.py --eval_train --checkpoint_dir="./runs/1546508380/checkpoints/
      
#### 这里1546508380需要修改, 根据自己的信息写


## 方法解释：
numpy库数组拼接np.concatenate官方文档详解
np.concatenate((a1, a2, …), axis=0)
- 传入的参数必须是一个多个数组的元组或者列表

- 另外需要指定拼接的方向，默认是 axis = 0，也就是说对0轴的数组对象进行纵向的拼接（纵向的拼接沿着axis= 1方向）；注：一般axis = 0，就是对该轴向的数组进行操作，操作方向是另外一个轴，即axis=1。

- 传入的数组必须具有相同的形状，这里的相同的形状可以满足在拼接方向axis轴上数组间的形状一致即可


tf.ConfigProto(
      log_device_placement=True,
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      allow_soft_placement=True)
      
sess = tf.Session(config=session_config)
  
- log_device_placement=True
    - 设置为True时，会打印出TensorFlow使用了那种操作

- inter_op_parallelism_threads=0
    - 设置线程一个操作内部并行运算的线程数，比如矩阵乘法，如果设置为０，则表示以最优的线程数处理

- intra_op_parallelism_threads=0
    - 设置多个操作并行运算的线程数，比如 c = a + b，d = e + f . 可以并行运算

- allow_soft_placement=True
    - 有时候，不同的设备，它的cpu和gpu是不同的，如果将这个选项设置成True，那么当运行设备不满足要求时，会自动分配GPU或者CPU。


### 关于json，pickle，itsdangerous中的loads\dumps的对比分析
查看点击：[参考网址](https://blog.csdn.net/Odyssues_lee/article/details/80921195)
