# iMDPath: Interpretable Multi-Task Digital Pathology Framework

## Overview

iMDPath is a comprehensive framework for digital pathology image analysis that integrates data augmentation, classification, and model interpretation. It addresses key challenges in medical image analysis including limited data diversity, model generalizability, and interpretability for cancer diagnosis.

### Key Components

* **iMDPath-Aug**: Vector Quantized Variational Autoencoder (VQVAE) for image enhancement

* **iMDPath-Pred**: Swin Transformer-based classification model

* **iMDPath-Exp**: Visualization module using FullGrad and occlusion sensitivity

## Features

### Data Augmentation

* VQVAE-based image enhancement

* Preservation of pathological features

* Improved downstream task performance

### Classification

* Swin Transformer architecture

* Transfer learning from ImageNet

* Multi-scale feature extraction

### Model Interpretability

* FullGrad visualization

* Occlusion sensitivity analysis

* Region-of-interest highlighting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iMDPath.git
cd iMDPath

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
pytorch-grad-cam>=1.3.7
```

## Dataset Structure

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── test/
    ├── class1/
    │   └── image5.jpg
    └── class2/
        └── image6.jpg
```

Dataset You can access the public dataset used for this project on Hugging Face at the following link:https://huggingface.co/chenqitao

## Usage

### Data Augmentation

```python
from src.augmentation import VQVAE
from src.utils.trainer import train_vqvae

# Initialize and train VQVAE
model = VQVAE()
train_vqvae(model, train_loader, num_epochs=50)
```

### Classification

```python
from src.classification import SwinClassifier
from src.utils.trainer import train_classifier

# Initialize and train classifier
model = SwinClassifier(num_classes=2)
train_classifier(model, train_loader, val_loader, num_epochs=60)
```

### Visualization

```python
from src.visualization import FullGradVisualizer
from src.utils.visualization import plot_heatmap

# Generate interpretability maps
visualizer = FullGradVisualizer(model)
heatmap = visualizer.generate_heatmap(image)
plot_heatmap(heatmap)
```

## Project Structure

```
iMDPath/
├── src/
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── vqvae.py
│   │   └── encoder_decoder.py
│   ├── classification/
│   │   ├── __init__.py
│   │   └── swin_classifier.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── fullgrad_vis.py
│   └── utils/
│       ├── __init__.py
│       ├── trainer.py
│       └── visualization.py
├── scripts/
│   ├── train_vqvae.py
│   ├── train_classifier.py
│   └── generate_visualizations.py
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md
```

## Model Performance

The framework has been evaluated on multiple datasets with the following results:

* Classification Accuracy: >90%

* AUC-ROC Score: >0.95

* Interpretability Score: High correlation with expert annotations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{imdpath2024,
  title={iMDPath: Interpretable Multi-Task Digital Pathology Framework},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

