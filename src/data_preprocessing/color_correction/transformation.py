import numpy as np
from PIL import Image
from .color_detection import return_colors_from_colorcard


def calculate_matrix_transform(image_path, first_image_colors=None):
    """
    Calculate the transformation matrix for color correction using linear algebra.

    If we have three vectors of reference colors (red, green, and blue) in RGB format,
    we can apply a linear transformation to adjust the entire image so these colors 
    match a desired set of reference colors using the transformation:

        v_after = A_transform @ v_before

    To determine `A_transform`, we solve:
        [v_R_ref, v_G_ref, v_B_ref] = A_transform @ [v_R, v_G, v_B]

    ### Reference Color Handling:
    - If `first_image_colors` is provided, it will be used for minimal correction across 
      all images, assuming the color card remains consistent across the dataset.
    - If not provided, the diagonal matrix will be used, performing standard normalization.

    Parameters:
    - image_path: Path to the image for color correction.
    - first_image_colors: Optional matrix with reference colors extracted from the first image.

    Returns:
    - A_transform: The calculated transformation matrix.
    """

    # Extract colors from the color card (assuming the function returns colors in RGB)
    M = return_colors_from_colorcard(image_path)  
    M_v_colors = np.array(M).T  # Transform to a matrix where columns represent R, G, B vectors

    # Compute the inverse of the color matrix for the transformation calculation
    M_v_colors_inv = np.linalg.inv(M_v_colors)

    # If a reference color matrix is provided, use it for minimal correction
    if first_image_colors is not None:
        # Ensure matrix is correctly formatted as a column matrix
        first_image_colors = np.array(first_image_colors).T  
        A_transform = first_image_colors @ M_v_colors_inv
    else:
        # Default: Diagonal matrix transformation
        A_transform = np.diag([1, 1, 1]) @ M_v_colors_inv  

    return A_transform


def apply_color_correction(img, A_transform):
    """
    Apply the calculated transformation matrix to an image for color correction.

    Each pixel in the image undergoes a linear transformation based on the matrix 
    calculated earlier. This ensures the extracted colors match the chosen reference set.

    Steps:
    1. Normalize the image (convert pixel values to [0,1]).
    2. Apply the transformation matrix to all pixels.
    3. Clip values to ensure no out-of-range results.
    4. Convert back to standard 8-bit RGB format.

    Parameters:
    - img: PIL Image object to be corrected.
    - A_transform: Transformation matrix obtained from `calculate_matrix_transform`.

    Returns:
    - corrected_img: PIL Image object with corrected colors.
    """
    # Convert the image to a NumPy array and normalize pixel values to [0,1]
    img_array = np.array(img, dtype=np.float32) / 255.0  

    # Reshape the image into a 2D array where each row is an RGB vector
    h, w, c = img_array.shape
    img_reshaped = img_array.reshape(-1, c)  

    # Apply the color transformation matrix to all pixels
    corrected_pixels = (A_transform @ img_reshaped.T).T  

    # Clip pixel values to ensure they stay within the [0, 1] range
    corrected_pixels = np.clip(corrected_pixels, 0, 1)

    # Reshape the corrected pixels back into the original image shape
    corrected_img_array = corrected_pixels.reshape(h, w, c)

    # Convert back to a PIL image with pixel values scaled back to [0, 255]
    corrected_img = Image.fromarray((corrected_img_array * 255).astype(np.uint8))
    return corrected_img
