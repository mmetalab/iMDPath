import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from src.visualization.fullgrad_vis import FullGradVisualizer
from src.utils.visualization import plot_heatmap
from src.classification.swin_classifier import SwinClassifier

# 初始化模型和可视化工具
model = SwinClassifier(num_classes=2, pretrained=True)
visualizer = FullGradVisualizer(model)

# 加载并预处理图像
transform = Compose([Resize((224, 224)), ToTensor()])
image = Image.open("path_to_image.jpg")
input_tensor = transform(image).unsqueeze(0).to("cuda")

# 生成可视化
heatmap = visualizer.generate_heatmap(input_tensor, target_class=1)
plot_heatmap(heatmap, title="FullGrad Heatmap")
