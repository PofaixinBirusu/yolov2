# yolov2算法复现

## 效果图
![1](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/1.png)
![2](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/2.png)
![3](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/3.png)
![4](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/4.png)
![5](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/5.png)

## 复现思路
直接用3个loss相加，不求均值，confidence loss和offset loss用总方，classify loss用BCELoss，模型是yolov2的darknet19，损失函数是yolov3的损失函数，相当于在yolov2的基础上改了一些。
![6](https://github.com/PofaixinBirusu/yolov2/blob/master/show-images/algorithm.jpg)
