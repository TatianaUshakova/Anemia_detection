# Anemia Detection using Deep Learning

**Warning**: project now in the process of being upload to GitHub and doesnt contain all the necessary components to be fully function

Anemia detection using deep learning, designed with the purpose of being deployed in rural areas of low income countries such as India where anemia-related complications, especially among pregnant women, are common and access to medical help is limited. So the goal of the project was to provede the help with anemia deagnostics in cases consulting medical professional or taking the blood test is unavailable.

The underlying idea, based on current literature, is that the skin colors of such body parts as tongue, palms, conjuctiva become more pale when a person have anemia, so it is possible to determine the anemic status of the patient based on pictures of such body parts.

The project consist of:

- Color correction from a color card (to insure correct color extraction from the body parts, independent on lightning)
- Image segmentation to select only the area of the body part
- General data cleaning and exploration (removing patients with missed info and incorrect images)
- Predictive models: classical regression on mean colors to neural networks (CNNs)

## Dataset Info
The original patient data used for model training involved Indian patients and remains private. For demonstration purposes, publicly available images similar to those used in the original project will be included. The images analyzed include the conjunctiva, nails, palms, and tongue, as paleness in these areas can indicate anemia.

## Project Structure
- `src/` - Source code including preprocessing and models
- `data/` - Will contains public images for demonstration only
- `notebooks/` - Will contain Jupyter notebooks for exploratory analysis
- `requirements.txt` - Python package dependencies

## Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/TatianaUshakova/Anemia_detection
   cd anemia-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the color correction pipeline if needed:**
   The color correction pipeline adjusts the colors in images using the color card.
   ```bash
   python -m data_preprocessing.color_correction.full_pipeline <folder_path> [<image_list>]
   ```
   - `<folder_path>`: Path to the folder containing the images.
   - `[<image_list>]`: Optional. Provide a list of image names, a CSV/Excel file, or leave empty to process all images in the folder.

4. **Run the image segmentation pipeline (to be uplooaded):**
   The image segmentation pipeline extracts regions of interest from the images.
   ```bash
   python -m data_preprocessing.image_segmentation.full_pipeline <folder_path> [<segmentation_parameters>]
   ```
   - `<folder_path>`: Path to the folder containing the images.
   - `[<segmentation_parameters>]`: Optional. Specify parameters for the segmentation algorithm (e.g., thresholds, masks).

5. **Run the prediction model pipeline (to be uploaded):**
   The RGB prediction model analyzes image RGB values and predicts specific outcomes.
   ```bash
   python -m data_preprocessing.rgb_prediction_model.predict <folder_path> [<model_parameters>]
   ```
   - `<folder_path>`: Path to the folder containing the images.
   - `[<model_parameters>]`: Optional. Specify parameters for the model, such as the path to a pre-trained model or specific prediction thresholds.


## Results (to be uploaded)
The project outputs include:
- Processed images with color correction and segmentation
- Trained models for anemia detection
- Evaluation metrics and performance reports

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


