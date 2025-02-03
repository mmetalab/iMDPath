import matplotlib.pyplot as plt

def plot_heatmap(heatmap, title="Heatmap", save_path=None):
    """
    Plot and optionally save a heatmap.

    Args:
        heatmap: Heatmap to be visualized.
        title: Title for the plot.
        save_path: Path to save the image (optional).
    """
    plt.imshow(heatmap, cmap="jet")
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.show()
