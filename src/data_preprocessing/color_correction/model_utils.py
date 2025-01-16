import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin
from PIL import Image
import numpy as np

def load_model():
    """
    Load the OWLVIT model and processor, setting the model to evaluation mode.
    Automatically uses CUDA if available.
    """
    model_name = "google/owlvit-base-patch32"
    global model
    global processor
    global device

    model = OwlViTForObjectDetection.from_pretrained(model_name)
    processor = OwlViTProcessor.from_pretrained(model_name)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

def image_preprocess(image_path):
    """
    Preprocesses an image for the OWLVIT model by resizing and normalizing.
    """
    image = Image.open(image_path).rotate(-90, expand=True)
    image_size = model.config.vision_config.image_size
    mixin = ImageFeatureExtractionMixin()
    image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0
    return input_image

def get_boxes_predictions(image_path, text_queries):
    """
    Perform object detection using the OWLVIT model on an image for the provided text queries.
    """
    image = Image.open(image_path).rotate(-90, expand=True)
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.max(outputs["logits"][0], dim=-1)  # Get max logits
    scores = torch.sigmoid(logits.values)
    labels = logits.indices
    boxes = outputs.pred_boxes[0]

    return scores, boxes, labels
