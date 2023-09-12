
# -----

import matplotlib.pyplot as plt

def plot_images(image_list, titles=None, cols=2, figsize=(12, 6)):
    """
    Plot a list of images using matplotlib.

    Args:
        image_list (list): List of image arrays to be plotted.
        titles (list, optional): List of titles for each image. If None, no titles will be shown.
        cols (int, optional): Number of columns for the grid of images.
        figsize (tuple, optional): Size of the figure (width, height) in inches.

    Returns:
        None
    """
    num_images = len(image_list)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if titles is None:
        titles = [None] * num_images

    for i, (image, title) in enumerate(zip(image_list, titles)):
        ax = axes[i // cols, i % cols]
        
        if len(image.shape) == 1:
            # If the image is 1D, treat it as a plot of a function
            ax.plot(image)
        elif len(image.shape) == 2:
            # If the image is 2D, display it as an image
            ax.imshow(image, cmap='gray')
        else:
            # Handle other dimensions (e.g., 3D RGB images)
            ax.imshow(image)
        
        ax.set_title(title)
        ax.axis('off')

    # Hide any remaining empty subplots
    for i in range(num_images, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
