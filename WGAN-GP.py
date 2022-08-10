# -*- coding:utf-8 -*-
# 作者：KKKC
# DCGAN Deep covlution GAN

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import datasets,transforms
from utils import gradient_penalty

cudnn.benchmark = True
# 定义参数
num_epoch = 40
lr = 1e-4
z_dimension = 100 # 初始噪声的维度以论文中的100为例
number_channel = 3 # 训练数据的通道数为3，根据数据集来修改
number_d_channel = 8 # Discriminator初始的通道数
number_g_channel = 8 # Generator初始的通道数
batch_size = 64
beta = 0.0
img_size = 64 # 论文中的图片大小
# WGAN
CRITIC_ITERATIONS = 3
L_GP= 10
transform = transforms.Compose([
    transforms.Resize([img_size,img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
# 使用ImageFolder来加载数据集，以celeb_dataset为例子
dataset = datasets.ImageFolder(root="../celeb_dataset/",transform=transform)
# dataset = datasets.MNIST(root="data",download=True,transform=transform)
data_loder = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

# Discriminator(Critic)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # input 1*64*64
            nn.Conv2d(number_channel,number_d_channel,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True), # x>0 => x ; x<0 =>0.1x
            # size: number_d_channel(ndf) * 32 * 32，没有池化层，使用stride来代替pool
            nn.Conv2d(number_d_channel,number_d_channel*2,4,2,1,bias=False),
            nn.InstanceNorm2d(number_d_channel*2),
            nn.LeakyReLU(0.2,inplace=True),
            # (ndf*2)*16*16
            nn.Conv2d(number_d_channel*2,number_d_channel*4,4,2,1,bias=False),
            nn.InstanceNorm2d(number_d_channel*4),
            nn.LeakyReLU(0.2,inplace=True),
            # (ndf*4)*8*8
            nn.Conv2d(number_d_channel*4,number_d_channel*8,4,2,1),
            nn.InstanceNorm2d(number_d_channel*8),
            nn.LeakyReLU(0.2,inplace=True),
            # (ndf*8)*4*4
            nn.Conv2d(number_d_channel*8,1,4,1,0,bias=False),
            # nn.Sigmoid() # 判别器的最后一层使用sigmoid激活函数
        )
    def forward(self,x):
        x = self.dis(x)
        return x # 去掉维度为1的维度

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input size: z_dimension
            # ConvTranspose2d：反卷积
            nn.ConvTranspose2d(z_dimension,number_g_channel*8,4,1,0,bias=False),
            nn.InstanceNorm2d(number_g_channel*8),
            nn.ReLU(True),

            # (ngf*16)*4*4
            nn.ConvTranspose2d(number_g_channel*8,number_g_channel*4,4,2,1,bias=False),
            nn.InstanceNorm2d(number_g_channel*4),
            nn.ReLU(True),

            # (ngf*8)*8*8
            nn.ConvTranspose2d(number_g_channel * 4, number_g_channel * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(number_g_channel * 2),
            nn.ReLU(True),

            # (ngf*4)*16*16
            nn.ConvTranspose2d(number_g_channel * 2, number_g_channel, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(number_g_channel),
            nn.ReLU(True),

            # (ngf*2)*32*32
            nn.ConvTranspose2d(number_g_channel, number_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self,x):
        x = self.gen(x)
        
        return x
# 初始化网络权重
def weight_init(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data,mean=0.0,std=0.02)

# 网络实例
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Critic = Discriminator().to(device)
Gen = Generator().to(device)
# 初始化权重
weight_init(Critic)
weight_init(Gen)
# optim
critic_optimizer = torch.optim.Adam(Critic.parameters(),lr=lr,betas=(beta,0.999))
gen_optimizer = torch.optim.Adam(Gen.parameters(),lr=lr,betas=(beta,0.999))
print('-----networt architecture-----')
print(Critic)
print(Gen)

# 生成噪声noise
fixed_noise = torch.randn(batch_size,z_dimension,1,1).to(device)
for epoch in range(num_epoch):
    for i,data in enumerate(data_loder):
        real = data[0].to(device)
        # real_label = torch.full((batch_size,),1.0).to(device)
        # fake_label = torch.full((batch_size,),0.0).to(device)
        # print(real_label.shape)
        for j in range(CRITIC_ITERATIONS):
            noise = torch.randn(batch_size,z_dimension,1,1).to(device)
            fake = Gen(noise) # 假图像
            critic_real = Critic(real).reshape(-1)
            critic_fake = Critic(fake).reshape(-1)
            gp = gradient_penalty(Critic,real,fake,device)
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + L_GP*gp
            print("{},Critic loss:{}".format(j,critic_loss))
            # 训练critic
            Critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

        # 训练Generator
        gen_fake = Critic(fake).reshape(-1)
        gen_loss = -torch.mean(gen_fake)
        Gen.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        if((i+1) % 10) == 0:
            fake = Gen(fixed_noise)
            torchvision.utils.save_image(
                fake.detach(),
                './WGAN-GP_fake_epoch{}.jpg'.format(epoch),
                normalize= True
            )
            print("[{}/{}][{}/{}] Loss_D:{} Loss_G:{}".format(epoch,num_epoch,i,len(data_loder),critic_loss,gen_loss))
        if(epoch+1)%5 ==0:
            Gen_dir = 'Gen'+str(epoch)+'.pkl'
            Dis_dir = 'Critic'+str(epoch)+'.pkl'
            torch.save(Gen.state_dict(),Gen_dir)
            torch.save(Critic.state_dict(),Dis_dir)

print("Training finish!!! save model")
torch.save(Gen.state_dict(), 'Gen_last.pkl')
torch.save(Critic.state_dict(), 'Critic_last.pkl')
print('save ok!!')