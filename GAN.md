## GAN

GAN（Generative Adversarial Net），生成对抗网络，其思想是同时训练两个网络，一个为判别器Discriminator，另一个是生成器Generator，以图片为例，判别器的任务就是识别出真的图片和假的图片，而生成器的任务就是生成图片来骗过判别器，同时训练两个网络，有种博弈的感觉，来达到一中平衡（纳什均衡）

<img src="https://pic3.zhimg.com/v2-5ca6a701d92341b8357830cc176fb8a3_1440w.jpg?source=172ae18b" style="zoom: 67%;" />

![img](https://upload-images.jianshu.io/upload_images/7749027-621c762069c0e21b.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/781/format/webp)

简言之，我们使用真实的训练数据和随机噪声生成的假图像来训练Discriminator，同时也更新Generator，循环往复训练。

## 网络的损失函数

- 对于D来说，要判别出真图像和假图像，而对于G来说要生成假图像来尽可能骗过D，GAN的两个网络是单独交替迭代进行训练的，先训练判别器，再训练生成器。

直接来看论文里面的两个损失函数

① 先看D的损失函数：

![img](https://upload-images.jianshu.io/upload_images/13326530-b6ffb3b6147a950d.png?imageMogr2/auto-orient/strip|imageView2/2/w/740/format/webp)

对于D来说，其中D(x)表示对真实的图片进行判别，D(G(z))表示对生成的假图片进行判别，D希望把所有的真图片都判定为真，而对于假图片都判定为假，即D(x)=>1，D(G(z))=>0，对于上式，则目标为最大化上式。

② G的损失函数

![img](https://upload-images.jianshu.io/upload_images/13326530-2ef7b257479e0a22.png?imageMogr2/auto-orient/strip|imageView2/2/w/627/format/webp)

对于G来说，要生成假图片来骗D，即D((z))=>1，整体要越小越好，即最小化上式

弄懂了两个网络的损失函数整个网络理解起来就比较容易了

## 网络的结构

GAN的网络的关键在于在D中使用全卷积网络来实现判别，在G中使用反卷积来生成图片。DCGAN论文中使用的图片的大小为64×64，最后G反卷得到的图像大小为64×64，通道数视任务而定。

`Generator:`

![img](https://img-blog.csdnimg.cn/20190313154200628.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FJX2dpcmw=,size_16,color_FFFFFF,t_70)

- G网使用ReLU作为激活函数，最后一层使用Tanh
- 去掉FC层，使用全卷积网络

`Discriminator:`

![img](https://img-blog.csdnimg.cn/20190313155110954.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FJX2dpcmw=,size_16,color_FFFFFF,t_70)

- D中取消所有的池化层，池化层用stride步长代替
- D网和G网均使用的BatchNormalization，但在最后一层不使用BatchNormalization，为了保证模型能够学习到正确的方差和均值
- D中使用LeakyReLU作为激活函数
- 优化器使用Adam

## 存在的缺点

DCGAN使用两个卷积神经网络，存在一些缺点：

- 不好训练，训练时间长；
- 参数设置的不好会梯度爆炸；
- 效果并不是很明显

## 训练的效果

`Mnist：`

训练了一个epoch，效果还行，第三个epoch梯度就爆了，图像花了。

<img src="G:\DCGAN\fake_sample_epoch0.jpg" alt="fake_sample_epoch0" style="zoom:50%;" />

`celeb_dataset:`

人脸数据集，训练的很慢，一个epoch训练了2个小时，使用了将近两个G显存<img src="C:\Users\HP\Desktop\cele_fake_sample_epoch0.jpg" alt="cele_fake_sample_epoch0" style="zoom:67%;" />

效果也不是很好，有个人样。

## 后续

使用WGAN来训练效果可能要更快更好

