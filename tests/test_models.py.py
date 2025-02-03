import torch
from src.augmentation.vqvae import VQVAE
from src.classification.swin_classifier import SwinClassifier

def test_vqvae():
    model = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512)
    input_tensor = torch.randn(1, 3, 128, 128)
    output, diff = model(input_tensor)
    assert output.shape == input_tensor.shape, "Output shape mismatch in VQVAE"

def test_classifier():
    model = SwinClassifier(num_classes=2, pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape[-1] == 2, "Output class mismatch in classifier"

if __name__ == "__main__":
    test_vqvae()
    test_classifier()
    print("All tests passed!")
