# Color Correction Project

This project performs automated color correction using deep learning and linear algebra techniques. It uses the OWL-ViT object detection model to identify color circles on a calibration card and applies a linear transformation for color adjustment.

## 📦 Project Structure
The `color_correction` module is organized as follows:
```
color_correction/
├── model_utils.py            # Loading the model and image preprocessing
├── color_detection.py        # Circle detection and geometric adjustments
├── transformation.py         # Color correction using linear algebra
├── visualization.py          # Visualization functions for debugging
├── __init__.py               # Import management for the module
```

## 🚀 Features
- **Deep Learning Integration:** Utilizes OWL-ViT for object detection.
- **Linear Color Transformation:** Applies linear algebra for color correction.
- **Image Preprocessing:** Rotates and resizes input images.
- **Visualization:** Tools for debugging and displaying results.

## 📊 How It Works
1. **Model Loading:** The `load_model()` function loads the OWL-ViT model and processor.
2. **Color Detection:** Circles on the calibration card are detected using object detection.
3. **Color Transformation:** A transformation matrix is calculated to normalize colors.
4. **Image Correction:** The matrix is applied to adjust the image colors.

## 📦 Installation
Ensure you have Python installed. To install the required dependencies:
```bash
pip install -r requirements.txt
```

## ✅ Usage
### Running in Python Script
```python
from color_correction import load_model, image_preprocess, calculate_matrix_transform, apply_color_correction, plot_original_vs_corrected

load_model()
img_path = "path/to/image.jpg"
img = image_preprocess(img_path)
A_transform = calculate_matrix_transform(img_path)
corrected_img = apply_color_correction(img, A_transform)
plot_original_vs_corrected(img, corrected_img)
```

## 📷 Example Visualizations
- **Detected Circles on the Color Card**
- **Original vs. Corrected Image Comparison**

## 📄 License
This project is licensed under the MIT License.

---
Feel free to contribute or raise issues for improvements!
