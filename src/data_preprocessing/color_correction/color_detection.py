import torch
import numpy as np
from .model_utils import get_boxes_predictions
from .model_utils import image_preprocess

def return_most_probable_box(target_label, scores, boxes, labels, text_queries):
  '''
  returns the most probable boxes for each label after getting several boxes of predicitions for each label in the text quiries
  (like 'green circle'), to detect the most probable circle position in the colorcard

  Usage:
  from data_preprocessing.color_correction.color_detection import return_most_probable_box
  text_queries = ['green circle', 'blue circle']
  scores, boxes, labels = get_boxes_predictions(image_path, text_queries)
  green_box = return_most_probable_box('green circle', scores, boxes, labels, text_queries)
  
  Can also be plotted with:
  from data_preprocessing.color_correction.visualization import plot_box_and_label
  from data_preprocessing.color_correction.model_utils import image_preprocess
  input_image = image_preprocess(image_path)
  plot_box_and_label(input_image, green_box, 'green circle')

  BEWARE: for the type of colorcard used sometimes the blue circle is identify with 'dark blue circle' when 'blue circle' gives incorrect predictions 
  '''
  label_index = text_queries.index(target_label)
  max_idx = torch.argmax(scores[labels == label_index]).item()
  return boxes[labels == label_index][max_idx]


def identify_red_box(blue_box, green_box):
    """
    Estimate the red circle's position and size based on the relative positions 
    and sizes of the blue and green circles.

    The model sometimes fails to detect the red circle due to:
    - Misidentification of body parts, such as the forehead mark in Indian datasets.
    - Red circle confusion with body regions instead of the color card.

    To correct this, a geometric approach is used:
    - Position: The red circle is located the same distance from the green circle as 
      the blue circle but in the opposite direction.
    - Size: The red circle's size is determined using perspective scaling, 
      ensuring the red circle is proportionally smaller than the green circle 
      in a similar ratio as the blue circle being larger.
    """
    # Unpack circle properties (center x, center y, width, height)
    cx_b, cy_b, w_b, h_b = blue_box
    cx_g, cy_g, w_g, h_g = green_box

    # Calculate the red circle's center using vector relationships
    # Red circle is symmetrically opposite to the blue circle from the green circle
    
    cx_r = 2 * cx_g - cx_b
    cy_r = 2 * cy_g - cy_b

    # Apply perspective correction for size estimation
    # The red circle is proportionally smaller than the green circle in the same ratio
    # as the blue circle is larger than the green circle.
    
    w_r = w_g**2 / w_b
    h_r = h_g**2 / h_b

    # Return the red circle's bounding box as a tensor
    red_box = torch.tensor([cx_r, cy_r, w_r, h_r])  
    return red_box


def get_average_color(box, image, scale=2):
    # Convert fractional box coordinates to pixel values
    height, width = image.shape[:2]
    cx, cy, w, h = box
    x_min, x_max, y_min, y_max = cx-w/(2*scale), cx+w/(2*scale), cy-h/(2*scale), cy+h/(2*scale) #select only central part of the box which def inside of circle

    # Scale fractional coordinates to pixel coordinates
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)

    # Ensure box coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # Crop the image to the box region
    cropped_region = image[y_min:y_max, x_min:x_max, :]

    # Calculate the mean color
    mean_colors = []
    for color in range(3):
      mean_colors.append(cropped_region[:,:,color].mean())

    return mean_colors


def return_colors_from_colorcard(image_path):
  '''
  we get the red, green, blue colors from colorcard by:
  1. Selecting green and blue circles by OWL ViT and getting the most probable of its predictions
  2. Identifying red circle by vector calculus
  3. Calculating the average colors by selecting the boxes that lays surely inside of the circles and calculate average of these areas
  '''
  text_queries = ['green circle', 'blue circle']
  scores, boxes, labels = get_boxes_predictions(image_path, text_queries)

  green_box = return_most_probable_box('green circle', scores, boxes, labels, text_queries)
  blue_box = return_most_probable_box('blue circle', scores, boxes, labels, text_queries)
  red_box = identify_red_box(blue_box, green_box)

  input_image = image_preprocess(image_path)

  red = get_average_color(red_box, input_image)
  green = get_average_color(green_box, input_image)
  blue = get_average_color(blue_box, input_image)

  return [red, green, blue]