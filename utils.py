# -*- coding:utf-8 -*-
# 作者：KKKC

import torch
import torch.nn as nn
# 梯度惩罚
def gradient_penalty(critic,real,fake,device='cpu'):
    # 获取信息
    batch_size, c ,h ,w = real.shape
    print(batch_size)
    alpha = torch.rand((batch_size,1,1,1)).repeat(1,c,h,w).to(device)
    mixed_image = alpha * real + (1-alpha) * fake
    mixed_score = critic(mixed_image)
    # 计算梯度
    gradient = torch.autograd.grad(
        inputs=mixed_image,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]
    gredient = gradient.view(gradient.shape[0],-1)
    # 计算二范数
    gradient_norm = gredient.norm(2,dim=1)
    gp = torch.mean((gradient_norm-1)**2) # 梯度损失
    return gp