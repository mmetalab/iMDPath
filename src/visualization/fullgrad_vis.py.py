import torch
import numpy as np
from typing import Optional, List, Union, Tuple
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn


class FullGradVisualizer:
    """
    A class for visualizing model predictions using FullGrad and occlusion sensitivity.

    Attributes:
        model: The PyTorch model to visualize
        target_layer: The target layer for FullGrad visualization
        device: The device to run computations on
        cam: FullGrad instance for generating activation maps
    """

    def __init__(self,
                 model: nn.Module,
                 target_layer: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the FullGradVisualizer.

        Args:
            model: PyTorch model for visualization
            target_layer: Target layer for FullGrad (if None, will attempt to get from model)
            device: Device to run computations on (if None, will use CUDA if available)
        """
        self.model = model
        self.target_layer = target_layer if target_layer else model.get_target_layer()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.cam = FullGrad(model=model, target_layers=[self.target_layer])

    def generate_heatmap(self,
                         input_tensor: torch.Tensor,
                         target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate heatmap for the input image using FullGrad.

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class for visualization (optional)

        Returns:
            Numpy array of the heatmap
        """
        input_tensor = input_tensor.to(self.device)

        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]

    def generate_occlusion_map(self,
                               input_tensor: torch.Tensor,
                               target_class: int,
                               occlusion_size: int = 5,
                               occlusion_stride: int = 4,
                               occlusion_value: float = 0.5,
                               weight_score: float = 2.0,
                               weight_fullgrad: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate occlusion sensitivity map combined with FullGrad visualization.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for visualization
            occlusion_size: Size of occlusion patch
            occlusion_stride: Stride for sliding the occlusion window
            occlusion_value: Value to use for occluded regions
            weight_score: Weight for classification score changes
            weight_fullgrad: Weight for FullGrad changes

        Returns:
            Tuple containing:
                - Normalized sensitivity map
                - Combined changes array
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        _, _, H, W = input_tensor.shape

        # Get original predictions and FullGrad map
        original_output = self.model(input_tensor)
        original_score = original_output[0, target_class].item()
        original_cam = self.generate_heatmap(input_tensor, target_class)

        # Initialize maps and lists for tracking changes
        sensitivity_map = np.zeros((H, W))
        score_changes = []
        fullgrad_changes = []

        # Apply occlusion and measure changes
        for y in range(0, H - occlusion_size + 1, occlusion_stride):
            for x in range(0, W - occlusion_size + 1, occlusion_stride):
                # Create occluded image
                occluded_image = input_tensor.clone()
                occluded_image[..., y:y + occlusion_size, x:x + occlusion_size] = occlusion_value

                # Get predictions and FullGrad map for occluded image
                with torch.no_grad():
                    occluded_output = self.model(occluded_image)
                occluded_score = occluded_output[0, target_class].item()
                occluded_cam = self.generate_heatmap(occluded_image, target_class)

                # Calculate changes
                score_change = original_score - occluded_score
                fullgrad_change = np.mean(np.abs(original_cam - occluded_cam))

                score_changes.append(score_change)
                fullgrad_changes.append(fullgrad_change)

                # Update sensitivity map
                sensitivity_map[y:y + occlusion_size, x:x + occlusion_size] += (
                        score_change * weight_score + fullgrad_change * weight_fullgrad
                )

        # Convert lists to arrays
        score_changes = np.array(score_changes)
        fullgrad_changes = np.array(fullgrad_changes)

        # Combine and normalize changes
        combined_changes = (weight_score * score_changes + weight_fullgrad * fullgrad_changes) / (
                weight_score + weight_fullgrad
        )

        # Normalize sensitivity map
        max_change = np.max(np.abs(sensitivity_map))
        if max_change > 0:
            sensitivity_map = sensitivity_map / max_change

        return sensitivity_map, combined_changes

    @staticmethod
    def visualize_results(image: torch.Tensor,
                          sensitivity_map: np.ndarray,
                          alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare visualization of the original image and sensitivity map overlay.

        Args:
            image: Original image tensor (C, H, W)
            sensitivity_map: Generated sensitivity map
            alpha: Transparency for the overlay

        Returns:
            Tuple containing:
                - Original image as numpy array
                - Overlay of sensitivity map on original image
        """
        # Convert image tensor to numpy array
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Create overlay
        overlay = np.copy(img_np)
        for c in range(3):  # Apply to each color channel
            overlay[..., c] = np.where(
                sensitivity_map > 0,
                img_np[..., c] * (1 - alpha) + sensitivity_map * alpha,
                img_np[..., c]
            )

        return img_np, overlay


def get_example_usage():
    """
    Example usage of the FullGradVisualizer class.
    """
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1]

    # Initialize visualizer
    visualizer = FullGradVisualizer(model, target_layer)

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Example of how to use the visualizer
    """
    # Load and preprocess your image
    image = Image.open('path_to_image.jpg')
    input_tensor = transform(image).unsqueeze(0)

    # Generate visualization
    sensitivity_map, changes = visualizer.generate_occlusion_map(
        input_tensor, 
        target_class=285  # Example class index
    )

    # Visualize results
    original, overlay = visualizer.visualize_results(
        input_tensor[0], 
        sensitivity_map
    )
    """


if __name__ == "__main__":
    get_example_usage()
