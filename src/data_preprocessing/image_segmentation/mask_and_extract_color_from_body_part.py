import pandas as pd
import os
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

def apply_mask(img_folder_path, mask_folder_path, img_name, rotate=True, png = True, exist_printing=False):
    '''
    in our dataset there are existing masks for part of the images, and they are rotated
    relative to the image. Hence the rotation option
    masks are named the same way as images, but with png extension (hence optional arg)
    returns masked image
    if image/mask doesnt exist returns None
    '''
    img_exist = os.path.exists(os.path.join(img_folder_path, img_name))
    if not img_exist:
        if exist_printing:
            print(f'Image doesnt exist: {img_name}')
        return None

    if rotate:
        img = Image.open(os.path.join(img_folder_path, img_name )).rotate(-90, expand=True)
    else:
        img = Image.open(os.path.join(img_folder_path, img_name ))
    
    #check if mask exist
    png_mask_exist = os.path.exists(os.path.join(mask_folder_path, img_name.replace('.jpg', '.png')))
    jpg_mask_exist = os.path.exists(os.path.join(mask_folder_path, img_name))

    if not (png_mask_exist or jpg_mask_exist):
        if exist_printing:
            print(f'mask doesnt exist for {img_name}')
        return None

    if png and png_mask_exist:
        mask = Image.open(os.path.join(mask_folder_path, img_name.replace('.jpg', '.png')))
    elif png:
        if exist_printing:
            print(f'png chosen but png mask doesnt exist for {img_name}')
        return None
    else:
        mask = Image.open(os.path.join(mask_folder_path, img_name))
    
    image_array = np.array(img)
    mask_array = np.array(mask)
    
    #handle differnt masks since some masks are colored, for example black-red
    mask_sum = mask_array.sum(axis=-1)  # Sum RGB channels
    binary_mask = (mask_sum > 0).astype(int)  # Normalize to 0s and 1s

    masked_image_array = image_array * binary_mask[:, :, None]  # Broadcast binary mask to 3 channels
    return masked_image_array.astype(np.uint8)


# Function to calculate statistics for RGB values
def calculate_rgb_statistics(masked_image_array):
    """
    Calculate average, std, skewness, and kurtosis for RGB values for non-black pixels.
    Passed masked image (image_array)
    To Do: make skew and kurtosis optional?
    return saverage, std, skewness, and kurtosis 

    The function is necessary as according to the literature the paleness of specific bodyparts 
    (such as tongue, conjuctiva, palms, nails) assessed through mean color and these statistics 
    is correlated to hemoglobin level and can be a prediction factor for anemia detection
    """
    # Identify non-black (non-masked) pixels
    non_black_mask = np.any(masked_image_array > 0, axis=-1)
    non_black_pixels = masked_image_array[non_black_mask]

    # Calculate statistics for each channel
    if non_black_pixels.size == 0:  # Handle empty masks
        return [np.nan] * 12  # 4 statistics for 3 channels

    stats = []
    for channel in range(3):  # R, G, B
        channel_data = non_black_pixels[:, channel]
        stats.extend([
            np.mean(channel_data),
            np.std(channel_data),
            skew(channel_data),
            kurtosis(channel_data)
        ])    
    return stats

def calculate_rgb_stats_for_df(df, img_folder_path, mask_folder_path, rotate=True, png = True):
    stats_list = []
    for i, row in df.iterrows():
        img_name = row['Images']
        masked_array = apply_mask(img_folder_path, mask_folder_path, img_name, rotate=rotate, png = png, exist_printing=False)
        if masked_array is not None:
            stats = calculate_rgb_statistics(masked_array)
            stats_list.append(stats)
        else:
            stats_list.append([np.nan] * 12)  # Append NaNs for missing masks or image

    # Add stats as new columns
    stats_cols = [
        "Mean_R", "Std_R", "Skew_R", "Kurt_R",
        "Mean_G", "Std_G", "Skew_G", "Kurt_G",
        "Mean_B", "Std_B", "Skew_B", "Kurt_B"
    ]
    stats_df = pd.DataFrame(stats_list, columns=stats_cols)
    df = pd.concat([df, stats_df], axis=1)
    df = df.dropna()#drop rows with missig stats = no masks or image
    return df

def debug_existing_masked_images(df, img_folder_path, mask_folder_path, debug_limit=5, rotate=True, png=True):
    """
    visualize N first existing masked images for debagging purposes: 
    sometimes pipeline doesnt work as expected and this helps catching this.

    Args:
    - df (pd.DataFrame): DataFrame containing image names in the 'Images' column.
    - img_folder_path (str): Path to the folder containing images.
    - mask_folder_path (str): Path to the folder containing masks.
    - debug_limit (int): Number of existing masked images to visualize.
    - rotate (bool): Whether to rotate the images.
    - png (bool): Whether to look for PNG masks.
    """
    count = 0
    for i, row in df.iterrows():
        if count >= debug_limit:
            break
        img_name = row['Images']
        masked_array = apply_mask(img_folder_path, mask_folder_path, img_name, rotate=rotate, png=png, exist_printing=True)
        if masked_array is not None:
            count += 1
            print(f"Masked Image for {img_name}:")
            plt.imshow(masked_array)
            plt.title(f"Masked Image for {img_name}")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(2)  # Pause for 2 seconds to view the image
            plt.close()
        else:
            print(f"Skipping {img_name}: Image or mask does not exist.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('img_folder_path', type = str, help="Path to the folder containing images.")
    parser.add_argument('mask_folder_path', type = str, help="Path to the folder containing masks of images.")
    parser.add_argument('df', type = str, help = 'Excel table with image names and hemoglobin')
    parser.add_argument('--rotate', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--png', type=lambda x: x.lower() == 'true', default=True)

    #optional fro debaging - printing masked images
    parser.add_argument('--debug', type=lambda x: x.lower() == 'true', help="Enable debugging to visualize existing masked images.")
    parser.add_argument('--debug_limit', type=int, default=5, help="Number of existing masked images to visualize (default: 5).")


    args = parser.parse_args()

    df = pd.read_excel(args.df)
    stats_df = calculate_rgb_stats_for_df(df, args.img_folder_path, args.mask_folder_path, args.rotate, args.png) 
    stats_df.to_excel("stats_rgb_data.xlsx", index=False)
    print("DataFrame with Stats of body-part colors saved to 'stats_rgb_data.xlsx'")
    
    # Debugging: Visualize masked images if debug is enabled
    if args.debug:
        debug_existing_masked_images(
            df,
            img_folder_path=args.img_folder_path,
            mask_folder_path=args.mask_folder_path,
            debug_limit=args.debug_limit,
            rotate=args.rotate,
            png=args.png
        )
