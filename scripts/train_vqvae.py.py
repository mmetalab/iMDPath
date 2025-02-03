import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.augmentation.vqvae import VQVAE
from src.utils.trainer import train_model

# 数据准备
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 初始化模型
vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=0.0002)
criterion = lambda recon, input, diff: ((recon - input) ** 2).mean() + diff

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae = train_model(vqvae, train_loader, epochs=50, criterion=criterion, optimizer=optimizer, device=device)
