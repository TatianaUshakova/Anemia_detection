import matplotlib.pyplot as plt

def plot_box_and_label(input_image, box, label_name):
    """
    Visualize the detected box with a label on the provided image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    cx, cy, w, h = box
    ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
            [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")

    ax.text(
        cx - w / 2,
        cy + h / 2 + 0.015,
        f"{label_name}",
        ha="left",
        va="top",
        color="red",
        bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"}
    )
    plt.show()

def plot_color_in_box(input_image, box, label_name, scale=2):
    """
    Visualize the area from which the average color is calculated.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    cx, cy, w, h = box
    ax.plot([cx-w/(2*scale), cx+w/(2*scale), cx+w/(2*scale), cx-w/(2*scale), cx-w/(2*scale)],
            [cy-h/(2*scale), cy-h/(2*scale), cy+h/(2*scale), cy+h/(2*scale), cy-h/(2*scale)], "r")

    ax.text(
        cx - w / 2,
        cy + h / 2 + 0.015,
        f"{label_name}",
        ha="left",
        va="top",
        color="red",
        bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"}
    )
    plt.show()

def plot_original_vs_corrected(original_image, corrected_image, close=False):
    """
    Display the original and corrected images side by side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(corrected_image)
    axes[1].set_title("Corrected Image")
    axes[1].axis('off')

    plt.tight_layout()
    if close:
        plt.show(block=False)
        plt.pause(2)  # Keep the image open for 2 seconds (adjustable) 
        plt.close()
    else:    
        plt.show()