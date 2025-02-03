import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.classification.swin_classifier import SwinClassifier
from src.utils.trainer import train_model

# 数据准备
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 初始化模型
classifier = SwinClassifier(num_classes=2, pretrained=True)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = train_model(classifier, train_loader, epochs=20, criterion=criterion, optimizer=optimizer, device=device)
