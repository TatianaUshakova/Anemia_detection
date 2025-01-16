import os
import sys
import pandas as pd
import numpy as np

from data_preprocessing.color_correction import (
    load_model, image_preprocess, calculate_matrix_transform, 
    apply_color_correction, plot_original_vs_corrected
)
from PIL import Image

def run_color_correction_pipeline(folder_path, image_list=None, first_image_path=None):
    """
    Runs the full color correction pipeline for a list of image paths provided via a CSV, Excel, list, or DataFrame.

    Parameters:
    - folder_path: Path to the folder containing the images.
    - image_list: Can be empty, a single image path, a list of image paths, a CSV/Excel file, or a Pandas/Numpy DataFrame containing image paths. If empty, color correction is performed on all the images in the folder.
    - first_image_path: Optional reference image for minimal correction mode.
    """
    load_model()

    # If image_list is empty, use all images in the folder
    if not image_list:
        image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    # Handling different types of inputs
    elif isinstance(image_list, str):
        if image_list.endswith(".csv"):
            df = pd.read_csv(image_list)
            image_paths = [os.path.join(folder_path, img) for img in df['image_name']]
        elif image_list.endswith(".xlsx"):
            df = pd.read_excel(image_list)
            image_paths = [os.path.join(folder_path, img) for img in df['image_name']]
        else:
            # Single image path
            image_paths = [os.path.join(folder_path, image_list)]
    elif isinstance(image_list, (list, np.ndarray)):
        image_paths = [os.path.join(folder_path, img) for img in image_list]
    elif isinstance(image_list, (pd.DataFrame, pd.Series)):
        image_paths = [os.path.join(folder_path, img) for img in image_list.iloc[:, 0].tolist()]
    else:
        raise ValueError("Image list must be empty, a single path, a CSV, Excel, list, or DataFrame containing image paths.")

    # Optional minimal correction mode based on the first image
    first_image_colors = None
    if first_image_path:
        first_image_colors = calculate_matrix_transform(first_image_path)

    # Process each image
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        if not os.path.exists(image_path):
            print(f"❌ Error: {image_path} not found. Skipping.")
            continue

        img = Image.open(image_path)
        input_image = image_preprocess(image_path)

        # Apply transformation and correction
        A_transform = calculate_matrix_transform(image_path, first_image_colors)
        corrected_img = apply_color_correction(img, A_transform)

        # Save the corrected image with the appropriate suffix and extension
        base, ext = os.path.splitext(image_path)
        corrected_img.save(f"{base}_corrected{ext}")
        plot_original_vs_corrected(img, corrected_img, close=True)

    print("\n✅ Color correction complete!")

# Example usage with command line input
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data_preprocessing.color_correction.full_pipeline <folder_path> [<image_list>]")
        sys.exit(1)

    folder_path = sys.argv[1]
    image_list = sys.argv[2] if len(sys.argv) > 2 else None
    run_color_correction_pipeline(folder_path, image_list)
