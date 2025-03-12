import math
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def load_idx_images(file_path):
    """
    从 IDX 文件中加载图像数据，并归一化到 [-1, 1]
    """
    with open(file_path, 'rb') as f:
        # 文件头包含：魔数（4字节）、图片数（4字节）、行数（4字节）、列数（4字节）
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取所有像素，dtype 为 uint8，每个像素占 1 字节
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        # 转换为 float32 并归一化到 [-1, 1]
        images = images.astype(np.float32)
        images = images / 127.5 - 1.0
    return images


class MNISTDataset(Dataset):
    def __init__(self, image_file):
        self.images = load_idx_images(image_file)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # 为了匹配网络输入要求，扩展通道维度（1, H, W）
        img = self.images[index]
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img)

train_images_path = 'train-images.idx3-ubyte'
train_dataset = MNISTDataset(train_images_path)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
T = 200  # 扩散总步数（可调，MNIST较简单，200步效果已足够）
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)  # (T,)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)


def forward_diffusion_sample(x0, t):
    """
    根据时间步 t 为 x0 添加噪声，生成 x_t
    :param x0: 原始图像 (B, 1, 28, 28)
    :param t: 时间步，形状 (B,)
    :return: x_t, noise
    """
    # 根据每个样本 t 索引 alpha_bar 值
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t]).view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

def get_sinusoidal_embedding(t, embedding_dim):
    """
    t: Tensor, shape (B,) 或 (B, 1)
    返回: Tensor, shape (B, embedding_dim)
    """
    # 保证t为float类型
    t = t.float().unsqueeze(1)  # (B, 1)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embedding_dim)
    if embedding_dim % 2 == 1:  # 若embedding_dim为奇数，则padding一维
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_embedding_dim=64):
        super(UNet, self).__init__()
        self.time_embedding_dim = time_embedding_dim

        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 将时间嵌入映射到瓶颈通道数（用于添加偏置）
        self.time_embed_bottleneck = nn.Sequential(
            nn.Linear(time_embedding_dim, base_channels * 4),
            nn.ReLU()
        )

        # 解码器部分
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        :param x: 噪声图像，形状 (B, in_channels, H, W)
        :param t: 时间步，形状 (B,)（整数张量）
        :return: 预测噪声，形状 (B, in_channels, H, W)
        """
        # 使用 sinusoidal 位置编码生成时间嵌入
        time_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)  # (B, time_embedding_dim)

        # 编码器
        e1 = self.enc1(x)  # (B, base_channels, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, base_channels*2, H/2, W/2)

        # 瓶颈部分
        b = self.bottleneck(self.pool2(e2))  # (B, base_channels*4, H/4, W/4)
        # 通过线性层将时间嵌入映射后扩展，并与b相加
        t_emb = self.time_embed_bottleneck(time_emb).unsqueeze(-1).unsqueeze(-1)  # (B, base_channels*4, 1, 1)
        b = b + t_emb

        # 解码器
        d2 = self.up2(b)  # (B, base_channels*2, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # 拼接，形状 (B, base_channels*4, H/2, W/2)
        d2 = self.dec2(d2)  # (B, base_channels*2, H/2, W/2)

        d1 = self.up1(d2)  # (B, base_channels, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, base_channels*2, H, W)
        d1 = self.dec1(d1)  # (B, base_channels, H, W)

        out = self.out_conv(d1)  # (B, in_channels, H, W)
        return out

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 15  # 训练周期，可适当增加

for epoch in range(epochs):
    for i, images in enumerate(train_loader):
        images = images.to(device)  # x0, 形状 (B, 1, 28, 28)
        batch_size = images.shape[0]
        # 随机为每个样本选择一个时间步 t ∈ [0, T-1]
        t = torch.randint(0, T, (batch_size,), device=device)
        # 生成带噪 x_t 和真实噪声
        x_t, noise = forward_diffusion_sample(images, t)
        # 模型预测噪声
        pred_noise = model(x_t, t)
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch} Step {i}, Loss: {loss.item():.4f}")

    # 每个 epoch 后，利用逆向过程采样生成一张图片
    with torch.no_grad():
        sample = torch.randn(1, 1, 28, 28).to(device)
        for t_inv in reversed(range(T)):
            t_tensor = torch.tensor([t_inv], device=device)
            pred_noise = model(sample, t_tensor)
            beta = betas[t_inv]
            alpha = alphas[t_inv]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            # 逆向更新公式（简化版）
            sample = (sample - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            if t_inv > 0:
                sample = sample + torch.sqrt(beta) * torch.randn_like(sample)
        generated_img = sample.clamp(-1, 1)
        plt.imshow((generated_img.cpu().squeeze() + 1) / 2, cmap='gray')
        plt.title(f"Epoch {epoch} Generated Sample")
        plt.axis('off')
        plt.savefig(f"generated_sample_epoch_{epoch}.png", bbox_inches='tight', pad_inches=0)

        plt.show()

print("训练完毕！")
for i in range(10):
        sample = torch.randn(1, 1, 28, 28).to(device)
        for t_inv in reversed(range(T)):
            t_tensor = torch.tensor([t_inv], device=device)
            pred_noise = model(sample, t_tensor)
            beta = betas[t_inv]
            alpha = alphas[t_inv]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            # 逆向更新公式（简化版）
            sample = (sample - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            if t_inv > 0:
                sample = sample + torch.sqrt(beta) * torch.randn_like(sample)
        generated_img = sample.clamp(-1, 1)
        plt.imshow((generated_img.cpu().detach().squeeze() + 1) / 2, cmap='gray')
        plt.title(f" result {i} Generated Sample")
        plt.axis('off')
        plt.savefig(f"result_generated_sample{i}.png", bbox_inches='tight', pad_inches=0)

        plt.show()