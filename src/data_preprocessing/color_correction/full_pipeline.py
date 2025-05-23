import os
import sys
import pandas as pd
import numpy as np

from data_preprocessing.color_correction.model_utils import load_model, image_preprocess
from data_preprocessing.color_correction.transformation import calculate_matrix_transform, apply_color_correction
from data_preprocessing.color_correction.visualization import plot_original_vs_corrected
from PIL import Image

def run_color_correction_pipeline(folder_path, image_list=None, output_folder = 'colorcorrected_images/',  first_image_path=None, printing=True):
    """
    Runs the full color correction pipeline for a list of image paths provided via a CSV, Excel, list, or DataFrame.

    Parameters:
    - folder_path: Path to the folder containing the images.
    - image_list: Can be empty, a single image path, a list of image paths, a CSV/Excel file, or a Pandas/Numpy DataFrame containing image paths. If empty, color correction is performed on all the images in the folder.
    - first_image_path: Optional reference image for minimal correction mode.
    """
    print(output_folder)
    load_model()

    # If image_list is empty, use all images in the folder
    if not image_list:
        image_list = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]


    first_image_colors = None
    if first_image_path:
        first_image_colors = calculate_matrix_transform(first_image_path)

    print(folder_path)
    print(output_folder)
    os.makedirs(os.path.join(folder_path, output_folder), exist_ok=True)
    
    # Process each image
    for image_name in image_list:
        try:
            image_path = os.path.join(folder_path, image_name)
            # Skip invalid paths
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            print(f"\nProcessing: {image_path}")
            img = Image.open(image_path)
            input_image = image_preprocess(image_path)

            # Apply transformation and correction
            A_transform = calculate_matrix_transform(image_path, first_image_colors)
            corrected_img = apply_color_correction(img, A_transform)

            # Save the corrected image in the output folder
            corrected_image_path = os.path.join(folder_path, output_folder, image_name)
            corrected_img.save(corrected_image_path)

            if printing:
                plot_original_vs_corrected(img, corrected_img, close=True)

        except np.linalg.LinAlgError as e:
            print(f"Skipping {image_path} due to singular matrix error: {e}")
        except Exception as e:
            print(f"Skipping {image_path} due to unexpected error: {e}")
    
    print("\nColor correction complete!")


if __name__ == "__main__":
    '''Usage: 
    python -m data_preprocessing.color_correction.full_pipeline <folder_path> [<image_list>]
    to process specific images in the image_list
    Or
    python -m data_preprocessing.color_correction.full_pipeline <folder_path>
    to process all the images in the folder
    '''

    folder_path = sys.argv[1]
    image_list = sys.argv[2:] if len(sys.argv) > 2 else None #process all the images if image_list is not specified
    run_color_correction_pipeline(folder_path, image_list, printing=False)
